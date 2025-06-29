import matplotlib
matplotlib.use('Agg')  # Set backend non-GUI untuk mencegah error tkinter
import os
import json
import numpy as np
import datetime
from django.urls import reverse
import io
import base64
import time
import logging
import sys
from django.conf import settings
from .models import Document, Result
from .word2vec_utils import Word2VecModel
from .document_processor import DocumentProcessor
from .ocr_processor import OCRProcessor
from .transformer_utils import TransformerEmbeddings
from .data_splitter import DataSplitter
from .evaluation_metrics import EvaluationMetrics
from .visualization_utils import Word2VecVisualizer
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Global search engine instance
_search_engine = None

def initialize_search_engine():
    """Initialize the search engine at Django startup"""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
        _search_engine.initialize_data_split()
        _search_engine.build_index()
        logging.info("Search engine initialized and index built")
    return _search_engine

def get_search_engine():
    """Get the global search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = initialize_search_engine()
    return _search_engine

class SearchEngine:
    def __init__(self):
        self.word2vec_model = Word2VecModel()
        self.document_processor = DocumentProcessor()
        self.ocr_processor = OCRProcessor()
        self.transformer = TransformerEmbeddings()
        self.data_splitter = DataSplitter(training_ratio=0.8)  # 80% training, 20% testing
        self.evaluation_metrics = EvaluationMetrics()
        self.visualizer = None  # Will be initialized after word2vec model is ready
        self.document_vectors = {}
        self.index_built = False
        self._last_search_data = None
        self._all_queries_results = []  # Store results from multiple queries for MAP/MRR calculation
    
    def initialize_data_split(self):
        """Initialize data splitting on startup"""
        try:
            training_count, testing_count = self.data_splitter.split_documents()
            split_info = self.data_splitter.get_split_info()
            
            logging.info(f"Data split initialized:")
            logging.info(f"  Training documents: {split_info['training']} ({split_info['training_ratio']:.1%})")
            logging.info(f"  Testing documents: {split_info['testing']} ({split_info['testing_ratio']:.1%})")
            logging.info(f"  Total documents: {split_info['total']}")
            
        except Exception as e:
            logging.error(f"Error initializing data split: {e}")
    
    def build_index(self, force=False):
        """Build search index using only testing data for search, but train transformer on training data"""
        if self.index_built and not force:
            logging.info("Index already built, skipping...")
            return
        
        start_time = time.time()
        logging.info("Building search index...")
        
        try:
            # Get training and testing documents separately
            training_documents = self.data_splitter.get_training_documents()
            testing_documents = self.data_splitter.get_testing_documents()
        
            # If no training documents, can't train models
            if not training_documents.exists():
                logging.warning("No training documents found for model training")
                self.index_built = True
                return
            
            # If no testing documents, can't build search index
            if not testing_documents.exists():
                logging.warning("No testing documents found for search index")
                self.index_built = True
                return
        
            # 1. Train Word2Vec model with training documents only
            training_document_tokens = self.document_processor.get_all_documents_tokens(training_documents)
            self.word2vec_model.train_model(training_document_tokens, force_train=force)
        
            # Initialize visualizer after Word2Vec model is ready
            self.visualizer = Word2VecVisualizer(self.word2vec_model)
        
            # 2. Train transformer embeddings on training data only
            training_doc_texts = {}
            for document in training_documents:
                full_path = document.document_path
                if not os.path.isabs(full_path):
                    full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                _, doc_text = self.document_processor.get_document_tokens(full_path)
                training_doc_texts[document.id] = doc_text
        
            # Pre-compute transformer embeddings for training documents (for model calibration)
            if training_documents:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    for document in training_documents:
                        executor.submit(
                            self.transformer.get_document_embedding,
                            document.id,
                            training_doc_texts.get(document.id, "")
                        )
        
            # Calibrate transformer threshold using training data
            self.transformer.recalibrate_threshold(list(training_documents), training_doc_texts)
        
            # 3. Build search index using only testing documents
            doc_vectors = {}
            testing_doc_texts = {}
        
            for document in testing_documents:
                # Get document path
                full_path = document.document_path
                if not os.path.isabs(full_path):
                    full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
            
                # Get document tokens and text
                doc_tokens, doc_text = self.document_processor.get_document_tokens(full_path)
            
                # Store document text for transformer
                testing_doc_texts[document.id] = doc_text
            
                # Get document vector using trained Word2Vec model
                doc_vector = self.word2vec_model.get_document_vector(document.id, doc_tokens)
                doc_vectors[document.id] = doc_vector
        
            # Save document vectors (only for testing documents)
            self.document_vectors = doc_vectors
            self.word2vec_model.save_vector_cache()
        
            # 4. Pre-compute transformer embeddings for testing documents only
            if testing_documents:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    for document in testing_documents:
                        executor.submit(
                            self.transformer.get_document_embedding,
                            document.id,
                            testing_doc_texts.get(document.id, "")
                        )
            self.transformer.save_cache()
        
            self.index_built = True
            training_count = training_documents.count()
            testing_count = testing_documents.count()
            logging.info(f"Search index built in {time.time() - start_time:.2f} seconds")
            logging.info(f"  Model trained on: {training_count} training documents")
            logging.info(f"  Search index built for: {testing_count} testing documents")
        
        except Exception as e:
            logging.error(f"Error building search index: {e}")

    def _calculate_dynamic_thresholds(self, similarities):
        """
        Calculate dynamic thresholds based on similarity distribution statistics
        No percentage limits - purely based on natural similarity ranges
        """
        if not similarities:
            return {
                'sangat_relevan': 0.75,
                'relevan': 0.55,
                'cukup_relevan': 0.35,
                'sedikit_relevan': 0.15
            }
        
        # Calculate statistical measures
        similarities_array = np.array(similarities)
        mean_sim = np.mean(similarities_array)
        std_sim = np.std(similarities_array)
        
        # Dynamic threshold calculation based on statistical distribution
        # Sangat Relevan: High similarity (mean + 1*std or above 0.7)
        sangat_relevan_threshold = max(0.7, min(0.9, mean_sim + 1.0 * std_sim))
        
        # Relevan: Above average similarity (mean + 0.5*std or above 0.5)
        relevan_threshold = max(0.5, min(sangat_relevan_threshold - 0.1, mean_sim + 0.5 * std_sim))
        
        # Cukup Relevan: Around average similarity (mean or above 0.3)
        cukup_relevan_threshold = max(0.3, min(relevan_threshold - 0.1, mean_sim))
        
        # Sedikit Relevan: Below average but not too low (mean - 0.5*std or above 0.1)
        sedikit_relevan_threshold = max(0.1, min(cukup_relevan_threshold - 0.1, mean_sim - 0.5 * std_sim))
        
        # Ensure meaningful gaps between thresholds
        if relevan_threshold >= sangat_relevan_threshold:
            relevan_threshold = sangat_relevan_threshold - 0.05
        if cukup_relevan_threshold >= relevan_threshold:
            cukup_relevan_threshold = relevan_threshold - 0.05
        if sedikit_relevan_threshold >= cukup_relevan_threshold:
            sedikit_relevan_threshold = cukup_relevan_threshold - 0.05
        
        return {
            'sangat_relevan': sangat_relevan_threshold,
            'relevan': relevan_threshold,
            'cukup_relevan': cukup_relevan_threshold,
            'sedikit_relevan': sedikit_relevan_threshold
        }

    def _categorize_by_natural_relevance(self, similarities_with_docs):
        """
        Categorize documents based on natural relevance without percentage constraints
        Each document gets the category it naturally deserves based on similarity score
        """
        if not similarities_with_docs:
            return []
        
        # Extract similarities for threshold calculation
        similarities = [sim for sim, _ in similarities_with_docs]
        
        # Calculate dynamic thresholds based on actual similarity distribution
        thresholds = self._calculate_dynamic_thresholds(similarities)
        
        # Categorize each document based on its actual similarity
        categorized_results = []
        
        for similarity, document in similarities_with_docs:
            # Natural categorization - no artificial limits
            if similarity >= thresholds['sangat_relevan']:
                category = "Sangat Relevan"
            elif similarity >= thresholds['relevan']:
                category = "Relevan"
            elif similarity >= thresholds['cukup_relevan']:
                category = "Cukup Relevan"
            elif similarity >= thresholds['sedikit_relevan']:
                category = "Sedikit Relevan"
            else:
                category = "Tidak Relevan"
            
            categorized_results.append((similarity, document, category))
        
        return categorized_results

    def search(self, ocr_text, ocr_id=None, search_start_time=None):
        """
        Natural relevance-based search without percentage constraints
        Documents are categorized based on their actual similarity scores
        """
        if search_start_time is None:
            search_start_time = time.time()

        if not self.index_built:
            self.build_index()

        try:
            # Only search in testing documents
            testing_documents = self.data_splitter.get_testing_documents()
            
            if not testing_documents.exists():
                logging.warning("No testing documents available for search")
                return [], None
            
            ocr_tokens = self.document_processor.preprocess_text(ocr_text)
            if not ocr_tokens:
                return [], None

            doc_texts = {}
            for document in testing_documents:
                full_path = document.document_path
                if not os.path.isabs(full_path):
                    full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                _, doc_text = self.document_processor.get_document_tokens(full_path)
                doc_texts[document.id] = doc_text

            # Use comprehensive transformer filtering with cached embeddings
            testing_docs_list = list(testing_documents)
            
            # Use moderate threshold for quality while ensuring comprehensive coverage
            original_threshold = self.transformer.threshold
            self.transformer.threshold = 0.05  # Moderate threshold for quality coverage
            
            filtered_indices = self.transformer.filter_documents(ocr_text, testing_docs_list, doc_texts)
            
            # Ensure we have sufficient documents for natural distribution
            min_docs_needed = min(100, len(testing_docs_list))  # At least 100 docs or all available
            if len(filtered_indices) < min_docs_needed:
                # Expand search to get more natural distribution
                self.transformer.threshold = 0.01
                filtered_indices = self.transformer.filter_documents(ocr_text, testing_docs_list, doc_texts)
            
            # Restore original threshold
            self.transformer.threshold = original_threshold
            
            filtered_documents = [testing_docs_list[i] for i in filtered_indices]
            
            query_vector = self.word2vec_model.get_query_vector(ocr_tokens)
            doc_vectors = {}

            # Calculate similarities for all filtered documents
            similarities_with_docs = []
            
            for document in filtered_documents:
                if document.id in self.document_vectors:
                    doc_vectors[document.id] = self.document_vectors[document.id]
                else:
                    full_path = document.document_path
                    if not os.path.isabs(full_path):
                        full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                    doc_tokens, _ = self.document_processor.get_document_tokens(full_path)
                    doc_vectors[document.id] = self.word2vec_model.get_document_vector(document.id, doc_tokens)

            # Calculate similarities in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    document.id: executor.submit(
                        self.word2vec_model.compute_similarity,
                        query_vector,
                        doc_vectors[document.id]
                    )
                    for document in filtered_documents
                }

                for document in filtered_documents:
                    similarity = futures[document.id].result()
                    similarities_with_docs.append((similarity, document))

            # Sort by similarity (highest first)
            similarities_with_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Apply natural relevance-based categorization
            categorized_results = self._categorize_by_natural_relevance(similarities_with_docs)
            
            # Create results and save to database
            results = []
            for similarity, document, category in categorized_results:
                result = Result(
                    id=document.id,
                    judul_artikel=document.document_name,
                    ekstrak_teks=f"{len(doc_texts.get(document.id, ''))} karakter yang ditemukan",
                    tokenisasi=json.dumps(self.document_processor.get_document_tokens(
                        os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(document.document_path))
                    )[0]),
                    kueri_ocr=json.dumps(ocr_tokens),
                    pelatihan_model="",
                    perhitungan_vektor_dokumen="",
                    perhitungan_kueri_ocr="",
                    perhitungan_kesamaan_kosinus=similarity,
                    keterangan=category
                )
                result.save()

                results.append({
                    'id': document.id,
                    'title': document.document_name,
                    'path': document.document_path,
                    'similarity': similarity,
                    'category': category
                })
            
            # Calculate total search time
            search_time = time.time() - search_start_time

            self._last_search_data = {
                'results': results.copy(),
                'ocr_tokens': ocr_tokens,
                'search_time': search_time,
                'doc_vectors': doc_vectors,
                'query_vector': query_vector,
                'doc_texts': doc_texts,
                'testing_docs_count': testing_documents.count()
            }
            
            # Store results for MAP/MRR calculation
            self._all_queries_results.append(results.copy())
            if len(self._all_queries_results) > 100:
                self._all_queries_results = self._all_queries_results[-100:]

            return results, None

        except Exception as e:
            logging.error(f"Error during search: {e}")
            return [], None
            
    def run_evaluation_async(self):
        """
        Run comprehensive evaluation with clean output format
        """
        try:
            # Pastikan matplotlib menggunakan backend non-GUI
            import matplotlib
            matplotlib.use('Agg')
            
            # Clear any existing matplotlib figures
            import matplotlib.pyplot as plt
            plt.close('all')
            if not hasattr(self, '_last_search_data') or not self._last_search_data:
                print("No search data available for evaluation")
                sys.stdout.flush()
                return

            data = self._last_search_data

            # Update Result objects with detailed information
            self._update_result_details(data)

            # Create UMAP visualization
            visualization_path = None
            if self.visualizer:
                try:
                    # Get top document tokens for visualization
                    top_document_tokens = self.visualizer.get_top_document_tokens(data['results'], data['doc_texts'])
                    visualization_path = self.visualizer.create_umap_visualization(
                        data['ocr_tokens'], 
                        top_document_tokens
                    )
                except Exception as e:
                    logging.error(f"Error creating visualization: {e}")

            # 1. ANALISIS KUERI OCR
            print("\n===== ANALISIS KUERI OCR =====")
            print(f"Jumlah token dalam kueri: {len(data['ocr_tokens'])}")
            print("Token kueri:")
            print(data['ocr_tokens'][:20])
            sys.stdout.flush()

            # 2. KONFIGURASI MODEL WORD2VEC
            print("\n===== KONFIGURASI MODEL WORD2VEC =====")
            print(f"Algoritma: Skipgram (sg=1)")
            print(f"Dimensi vektor: {self.word2vec_model.model.vector_size}")
            print(f"Window size: {self.word2vec_model.model.window}")
            print(f"Min count: {self.word2vec_model.model.min_count}")
            print(f"Workers: {self.word2vec_model.model.workers}")
            print(f"Vocabulary size: {len(self.word2vec_model.model.wv.key_to_index):,}")
            sys.stdout.flush()

            # 3. KONFIGURASI MODEL TRANSFORMER
            transformer_config = self.transformer.get_config_info()
            print("\n===== KONFIGURASI MODEL TRANSFORMER =====")
            print(f"Model : {transformer_config['model_name']}")
            print(f"Dimensi embedding : {transformer_config['embedding_dim']}")
            print(f"Multilangual : {transformer_config['multilingual']}")
            print(f"Cache size : {transformer_config['cache_size']}")
            print(f"Threshold filtering: {transformer_config['threshold']}")
            sys.stdout.flush()

            # 4. VISUALISASI VEKTOR WORD2VEC (UMAP)
            print("\n===== VISUALISASI VEKTOR WORD2VEC (UMAP) =====")
            if visualization_path:
                print(visualization_path)
            else:
                print("Visualisasi tidak dapat dibuat")
            sys.stdout.flush()

            # 5. ANALISIS KINERJA PENCARIAN
            print("\n===== ANALISIS KINERJA PENCARIAN =====")
            print(f"Waktu pencarian: {data['search_time']:.2f} detik")
            sys.stdout.flush()

            # 6. EVALUASI HASIL PENCARIAN DOKUMEN
            print("\n===== EVALUASI HASIL PENCARIAN DOKUMEN =====")
            
            # Count documents by category
            categories = {
                'Sangat Relevan': 0,
                'Relevan': 0,
                'Cukup Relevan': 0,
                'Sedikit Relevan': 0,
                'Tidak Relevan': 0
            }

            for result in data['results']:
                categories[result['category']] += 1

            # Print category counts
            print(f"Sangat Relevan: {categories['Sangat Relevan']}")
            print(f"Relevan: {categories['Relevan']}")
            print(f"Cukup Relevan: {categories['Cukup Relevan']}")
            print(f"Sedikit Relevan: {categories['Sedikit Relevan']}")
            print(f"Tidak Relevan: {categories['Tidak Relevan']}")
            print(f"Total dokumen hasil pencarian: {len(data['results'])}")
            sys.stdout.flush()

            # 7. METRIK EVALUASI SISTEM
            print("\n==== METRIK EVALUASI SISTEM ====")
            
            # Calculate comprehensive metrics
            metrics = self.evaluation_metrics.calculate_comprehensive_metrics(
                data['results'], 
                precision_k=10,
                use_recall_without_k=True
            )
            
            # Calculate MAP and MRR from multiple queries if available
            map_score = self.evaluation_metrics.calculate_map_at_k(self._all_queries_results, k=10)
            mrr_score = self.evaluation_metrics.calculate_mrr_at_k(self._all_queries_results, k=10)

            print(f"Presisi@10: {metrics['precision']:.4f}")
            print(f"Recall (tanpa @K): {metrics['recall']:.4f}")
            print(f"F1-score (adaptif): {metrics['f1_score']:.4f}")
            print(f"MAP@10: {map_score:.4f}")
            print(f"MRR@10: {mrr_score:.4f}")
            print(f"Normalized Discounted Cumulative Gain (NDCG@10): {metrics['ndcg']:.4f}")

            # Force flush all output
            sys.stdout.flush()
            sys.stderr.flush()

        except Exception as e:
            print(f"\nERROR in run_evaluation_async: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    def _update_result_details(self, data):
        """Update Result objects with detailed information after search is complete"""
        try:
            # Get display tokens
            for result_data in data['results']:
                doc_id = result_data['id']
            
                try:
                    # Get the document
                    document = Document.objects.get(id=doc_id)
                
                    # Get document path
                    full_path = document.document_path
                    if not os.path.isabs(full_path):
                        full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                
                    # Get document tokens
                    doc_tokens, _ = self.document_processor.get_document_tokens(full_path)
                
                    # Get display tokens
                    display_tokens = set(data['ocr_tokens'][:10] + doc_tokens[:10])
                    model_vectors = self.word2vec_model.get_model_vectors(display_tokens)
                
                    # Format vectors for display - show all dimensions
                    model_vectors_str = ""
                    for word, vec in model_vectors.items():
                        if word in self.word2vec_model.model.wv:
                            # Show all dimensions
                            full_vec = self.word2vec_model.model.wv[word].tolist()
                            model_vectors_str += f'"{word}" → {full_vec}\n'
                
                    # Update the Result object with detailed information
                    try:
                        result = Result.objects.get(id=doc_id)
                        result.pelatihan_model = model_vectors_str
                        result.perhitungan_vektor_dokumen = str(data['doc_vectors'][doc_id].tolist())
                        result.perhitungan_kueri_ocr = str(data['query_vector'].tolist())
                        result.save()
                    except Result.DoesNotExist:
                        # Result might have been deleted
                        pass
                    
                except Document.DoesNotExist:
                    # Document might have been deleted
                    pass
                
        except Exception as e:
            logging.error(f"Error updating result details: {e}")
    
    def add_new_document(self, document_id):
        """Add new document to testing set and add to search index"""
        try:
            # Ensure document is in testing set
            self.data_splitter.add_new_document_to_testing(document_id)
        
            # Get the new document
            document = Document.objects.get(id=document_id)
        
            # Get document path
            full_path = document.document_path
            if not os.path.isabs(full_path):
                full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
        
            # Get document tokens and text
            doc_tokens, doc_text = self.document_processor.get_document_tokens(full_path)
        
            # Add document vector to index using existing trained Word2Vec model
            doc_vector = self.word2vec_model.get_document_vector(document.id, doc_tokens)
            self.document_vectors[document.id] = doc_vector
        
            # Add transformer embedding for the new document immediately
            self.transformer.get_document_embedding(document.id, doc_text)
        
            # Save updated caches
            self.word2vec_model.save_vector_cache()
            self.transformer.save_cache()
        
            logging.info(f"New document {document_id} added to testing set and search index")
            logging.info(f"Transformer embedding created for document {document_id}")
        
        except Exception as e:
            logging.error(f"Error adding new document {document_id}: {e}")

    def remove_document_from_index(self, document_id):
        """Remove document from search index when deleted"""
        try:
            # Remove from document vectors
            if document_id in self.document_vectors:
                del self.document_vectors[document_id]
        
            # Clear caches
            self.word2vec_model.clear_document_vector(document_id)
            self.transformer.clear_document_embedding(document_id)
        
            # Save updated caches
            self.word2vec_model.save_vector_cache()
            self.transformer.save_cache()
        
            logging.info(f"Document {document_id} removed from search index")
            logging.info(f"Transformer embedding cleared for document {document_id}")
        
        except Exception as e:
            logging.error(f"Error removing document {document_id} from index: {e}")
