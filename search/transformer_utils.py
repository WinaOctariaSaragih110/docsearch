import os
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import pickle
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class TransformerEmbeddings:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', cache_path=None):
        """Initialize transformer model for fast initial filtering"""
        self.cache_path = cache_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    'static', 'models', 'transformer_cache.pkl')
        self.embedding_cache = {}
        self.model = None
        self.model_name = model_name
        self.threshold = 0.2  # Default threshold for filtering
        
        try:
            # Load the model
            self.model = SentenceTransformer(model_name)
            logging.info(f"Loaded transformer model: {model_name}")
            
            # Load cache if exists
            self.load_cache()
        except Exception as e:
            logging.error(f"Error loading transformer model: {e}")
            logging.info("Transformer-based filtering will be disabled")
    
    def load_cache(self):
        """Load embedding cache from disk"""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logging.info(f"Loaded transformer cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            logging.error(f"Error loading transformer cache: {e}")
            self.embedding_cache = {}
    
    def save_cache(self):
        """Save embedding cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logging.info(f"Saved transformer cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            logging.error(f"Error saving transformer cache: {e}")
    
    def get_document_embedding(self, doc_id, text):
        """Get embedding for a document with caching"""
        if self.model is None:
            return None
            
        # Return cached embedding if available
        cache_key = f"doc_{doc_id}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Generate embedding
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logging.error(f"Error generating document embedding: {e}")
            return None
    
    def get_query_embedding(self, text):
        """Get embedding for a query"""
        if self.model is None:
            return None
            
        try:
            return self.model.encode(text, show_progress_bar=False)
        except Exception as e:
            logging.error(f"Error generating query embedding: {e}")
            return None
    
    def compute_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
            
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def filter_documents(self, query_text, documents, doc_texts, threshold=None):
        """Filter documents using transformer embeddings for fast initial filtering"""
        if threshold is None:
            threshold = self.threshold
            
        if self.model is None:
            # If transformer model is not available, return all documents
            return list(range(len(documents)))
            
        # Get query embedding
        query_embedding = self.get_query_embedding(query_text)
        if query_embedding is None:
            return list(range(len(documents)))
        
        # Get document embeddings in parallel
        doc_embeddings = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i, doc in enumerate(documents):
                futures.append(executor.submit(
                    self.get_document_embedding, 
                    doc.id, 
                    doc_texts.get(doc.id, "")
                ))
            
            for i, future in enumerate(futures):
                doc_embeddings[documents[i].id] = future.result()
        
        # Calculate similarities and filter
        similarities = []
        for i, doc in enumerate(documents):
            if doc_embeddings[doc.id] is not None:
                sim = self.compute_similarity(query_embedding, doc_embeddings[doc.id])
                similarities.append((i, sim))
        
        # Filter documents above threshold
        filtered_indices = [idx for idx, sim in similarities if sim >= threshold]
        
        # If no documents pass the threshold, return top 5
        if not filtered_indices and similarities:
            similarities.sort(key=lambda x: x[1], reverse=True)
            filtered_indices = [idx for idx, _ in similarities[:5]]
        
        return filtered_indices
    
    def clear_document_embedding(self, doc_id):
        """Remove document embedding from cache when document is deleted"""
        cache_key = f"doc_{doc_id}"
        if cache_key in self.embedding_cache:
            del self.embedding_cache[cache_key]
            self.save_cache()
    
    def rebuild_embeddings_for_documents(self, documents, doc_texts):
        """Rebuild embeddings for specific documents"""
        if self.model is None:
            return
            
        logging.info(f"Rebuilding transformer embeddings for {len(documents)} documents...")
        
        try:
            # Clear existing embeddings for these documents
            for doc in documents:
                cache_key = f"doc_{doc.id}"
                if cache_key in self.embedding_cache:
                    del self.embedding_cache[cache_key]
        
            # Rebuild embeddings in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for doc in documents:
                    doc_text = doc_texts.get(doc.id, "")
                    futures.append(executor.submit(
                        self.get_document_embedding, 
                        doc.id, 
                        doc_text
                    ))
            
            # Wait for all embeddings to complete
            for future in futures:
                future.result()
        
            # Save updated cache
            self.save_cache()
            logging.info(f"Successfully rebuilt embeddings for {len(documents)} documents")
        
        except Exception as e:
            logging.error(f"Error rebuilding embeddings: {e}")
    
    def recalibrate_threshold(self, training_documents, training_doc_texts):
        """Recalibrate threshold berdasarkan distribusi similarity dari data training"""
        if self.model is None or not training_documents:
            return
        
        print("[CONSISTENCY] Mengkalibrasi threshold transformer menggunakan data training...")
        
        try:
            # Sample beberapa query untuk kalibrasi dari training documents
            sample_queries = []
        
            # Extract sample queries from training documents
            for doc in training_documents[:5]:  # Use first 5 training documents as sample queries
                doc_text = training_doc_texts.get(doc.id, "")
                if doc_text:
                    # Extract first few sentences as sample query
                    sentences = doc_text.split('.')[:2]  # First 2 sentences
                    query = '. '.join(sentences).strip()
                    if len(query) > 20:  # Only use if query is meaningful
                        sample_queries.append(query)
        
            # If no good queries from documents, use default queries
            if not sample_queries:
                sample_queries = [
                    "transportasi umum",
                    "pendidikan tinggi", 
                    "pembangunan infrastruktur",
                    "kebijakan pemerintah",
                    "teknologi informasi"
                ]
        
            all_similarities = []
        
            for query in sample_queries:
                query_embedding = self.get_query_embedding(query)
                if query_embedding is not None:
                    for doc in training_documents:
                        doc_embedding = self.get_document_embedding(doc.id, training_doc_texts.get(doc.id, ""))
                        if doc_embedding is not None:
                            sim = self.compute_similarity(query_embedding, doc_embedding)
                            all_similarities.append(sim)
        
            if all_similarities:
                # Hitung statistik distribusi
                mean_sim = np.mean(all_similarities)
                std_sim = np.std(all_similarities)
            
                # Set threshold sebagai mean - 0.5*std untuk memfilter dokumen yang terlalu tidak relevan
                new_threshold = max(0.1, mean_sim - 0.5 * std_sim)
            
                print(f"[CONSISTENCY] Threshold lama: {self.threshold:.3f}")
                print(f"[CONSISTENCY] Threshold baru: {new_threshold:.3f}")
                print(f"[CONSISTENCY] Mean similarity: {mean_sim:.3f}, Std: {std_sim:.3f}")
                print(f"[CONSISTENCY] Kalibrasi menggunakan {len(training_documents)} dokumen training")
            
                self.threshold = new_threshold
            
        except Exception as e:
            print(f"[ERROR] Gagal mengkalibrasi threshold: {e}")
            
    def get_config_info(self):
        """Get configuration information for the transformer model"""
        if self.model is None:
            return {
                "model_name": "Not loaded",
                "embedding_dim": 0,
                "multilingual": False,
                "cache_size": 0,
                "threshold": self.threshold
            }
            
        return {
            "model_name": self.model_name,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            "multilingual": "multilingual" in self.model_name.lower(),
            "cache_size": len(self.embedding_cache),
            "threshold": self.threshold
        }
