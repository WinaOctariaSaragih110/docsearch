import random
import logging
from django.db import transaction
from .models import Document

class DataSplitter:
    def __init__(self, training_ratio=0.8):
        """
        Initialize data splitter
        training_ratio: Rasio data training (0.8 = 80% training, 20% testing)
        """
        self.training_ratio = training_ratio
        
    def split_documents(self, force_resplit=False):
        """
        Split documents into training and testing sets
        force_resplit: Force re-splitting even if already split
        """
        try:
            # Check if documents are already split
            total_docs = Document.objects.count()
            training_docs = Document.objects.filter(is_training=True).count()
            
            if training_docs > 0 and not force_resplit:
                logging.info(f"Documents already split: {training_docs} training, {total_docs - training_docs} testing")
                return training_docs, total_docs - training_docs
            
            if total_docs == 0:
                logging.warning("No documents found to split")
                return 0, 0
            
            # Calculate split sizes
            training_size = int(total_docs * self.training_ratio)
            testing_size = total_docs - training_size
            
            logging.info(f"Splitting {total_docs} documents: {training_size} training, {testing_size} testing")
            
            # Get all documents and shuffle them
            all_documents = list(Document.objects.all())
            random.seed(42)  # For reproducible results
            random.shuffle(all_documents)
            
            # Split documents
            with transaction.atomic():
                # Reset all documents to testing first
                Document.objects.all().update(is_training=False)
                
                # Set first training_size documents as training
                training_docs = all_documents[:training_size]
                training_ids = [doc.id for doc in training_docs]
                
                Document.objects.filter(id__in=training_ids).update(is_training=True)
            
            logging.info(f"Successfully split documents: {training_size} training, {testing_size} testing")
            return training_size, testing_size
            
        except Exception as e:
            logging.error(f"Error splitting documents: {e}")
            return 0, 0
    
    def get_training_documents(self):
        """Get all training documents"""
        return Document.objects.filter(is_training=True)
    
    def get_testing_documents(self):
        """Get all testing documents"""
        return Document.objects.filter(is_training=False)
    
    def add_new_document_to_testing(self, document_id):
        """Add newly added document to testing set"""
        try:
            Document.objects.filter(id=document_id).update(is_training=False)
            logging.info(f"Document {document_id} added to testing set")
            
            # Log current split after adding new document
            split_info = self.get_split_info()
            logging.info(f"Current split after adding document {document_id}:")
            logging.info(f"  Training: {split_info['training']}, Testing: {split_info['testing']}")
            
        except Exception as e:
            logging.error(f"Error adding document {document_id} to testing set: {e}")

    def ensure_document_in_testing(self, document_id):
        """Ensure a specific document is in testing set (for deleted documents that are re-added)"""
        try:
            document = Document.objects.get(id=document_id)
            if document.is_training:
                document.is_training = False
                document.save()
                logging.info(f"Document {document_id} moved from training to testing set")
            else:
                logging.info(f"Document {document_id} already in testing set")
        except Document.DoesNotExist:
            logging.error(f"Document {document_id} not found")
        except Exception as e:
            logging.error(f"Error ensuring document {document_id} in testing set: {e}")
    
    def get_split_info(self):
        """Get information about current split"""
        total_docs = Document.objects.count()
        training_docs = Document.objects.filter(is_training=True).count()
        testing_docs = total_docs - training_docs
        
        return {
            'total': total_docs,
            'training': training_docs,
            'testing': testing_docs,
            'training_ratio': training_docs / total_docs if total_docs > 0 else 0,
            'testing_ratio': testing_docs / total_docs if total_docs > 0 else 0
        }
