import os
from django.apps import AppConfig
import logging

class SearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'search'
    
    def ready(self):
        """
        This method is called when Django starts.
        Initialize the search engine here to build the Word2Vec model once.
        """
        # Hanya inisialisasi pada proses utama, bukan pada autoreloader
        # Cek apakah ini adalah proses utama Django, bukan proses autoreloader
        if not os.environ.get('RUN_MAIN'):
            return
            
        # Import here to avoid circular imports
        from .search_engine import initialize_search_engine
        
        # Initialize search engine when Django starts
        try:
            initialize_search_engine()
            logging.info("Search engine initialized during Django startup")
        except Exception as e:
            logging.error(f"Error initializing search engine during startup: {e}")
