import matplotlib
matplotlib.use('Agg')  # Set backend non-GUI sebelum import pyplot
import matplotlib.pyplot as plt
import os
import numpy as np
import umap
from sklearn.decomposition import PCA
import logging
from django.conf import settings
import matplotlib.patches as mpatches

class Word2VecVisualizer:
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model
        self.save_path = os.path.join(settings.BASE_DIR, 'static', 'images')
        
        # Ensure directory exists
        os.makedirs(self.save_path, exist_ok=True)
        
        # Define semantic categories for Indonesian words
        self.word_categories = {
            'Transportasi': {
                'words': ['transportasi', 'kendaraan', 'jalan', 'raya', 'mobil', 'motor', 'bus', 'kereta', 'api', 'pesawat', 'kapal', 'laut', 'udara', 'darat', 'angkutan', 'umum', 'terminal', 'stasiun', 'bandara', 'pelabuhan', 'tol', 'jembatan', 'flyover', 'underpass', 'traffic', 'lalu', 'lintas', 'parkir', 'garasi', 'bengkel'],
                'color': '#FF6B6B'
            },
            'Pemerintahan': {
                'words': ['pemerintah', 'pemerintahan', 'negara', 'republik', 'indonesia', 'presiden', 'menteri', 'gubernur', 'bupati', 'walikota', 'camat', 'lurah', 'kepala', 'desa', 'daerah', 'provinsi', 'kabupaten', 'kota', 'kecamatan', 'kelurahan', 'instansi', 'lembaga', 'departemen', 'kementerian', 'dinas', 'badan', 'kantor', 'pelayanan', 'publik', 'administrasi'],
                'color': '#4ECDC4'
            },
            'Hukum & Regulasi': {
                'words': ['hukum', 'undang', 'peraturan', 'perda', 'kepres', 'keppres', 'permendagri', 'permen', 'sk', 'keputusan', 'surat', 'edaran', 'instruksi', 'regulasi', 'aturan', 'ketentuan', 'pasal', 'ayat', 'bab', 'bagian', 'pengadilan', 'hakim', 'jaksa', 'polisi', 'advokat', 'notaris', 'hak', 'kewajiban', 'sanksi', 'denda'],
                'color': '#45B7D1'
            },
            'Pendidikan': {
                'words': ['pendidikan', 'sekolah', 'universitas', 'perguruan', 'tinggi', 'kampus', 'siswa', 'mahasiswa', 'guru', 'dosen', 'pengajar', 'belajar', 'mengajar', 'kuliah', 'kelas', 'ruang', 'perpustakaan', 'laboratorium', 'praktikum', 'ujian', 'tes', 'nilai', 'rapor', 'ijazah', 'diploma', 'sarjana', 'magister', 'doktor', 'beasiswa', 'kurikulum'],
                'color': '#96CEB4'
            },
            'Infrastruktur & Proyek': {
                'words': ['infrastruktur', 'pembangunan', 'konstruksi', 'proyek', 'gedung', 'bangunan', 'rumah', 'perumahan', 'kompleks', 'mall', 'pasar', 'toko', 'kantor', 'pabrik', 'industri', 'jembatan', 'dam', 'bendungan', 'irigasi', 'drainase', 'saluran', 'air', 'listrik', 'pln', 'telkom', 'internet', 'wifi', 'tower', 'antena'],
                'color': '#FFEAA7'
            },
            'Kesehatan & Lingkungan': {
                'words': ['kesehatan', 'rumah', 'sakit', 'puskesmas', 'dokter', 'perawat', 'obat', 'vaksin', 'imunisasi', 'covid', 'pandemi', 'virus', 'bakteri', 'penyakit', 'sehat', 'lingkungan', 'sampah', 'limbah', 'polusi', 'udara', 'air', 'tanah', 'hutan', 'pohon', 'taman', 'hijau', 'konservasi', 'daur', 'ulang'],
                'color': '#DDA0DD'
            },
            'Teknologi & Sistem': {
                'words': ['teknologi', 'sistem', 'komputer', 'laptop', 'handphone', 'smartphone', 'aplikasi', 'software', 'hardware', 'program', 'coding', 'website', 'online', 'digital', 'elektronik', 'robot', 'otomatis', 'canggih', 'modern', 'inovasi', 'penelitian', 'riset', 'sains', 'ilmu', 'pengetahuan', 'data', 'informasi', 'database', 'server', 'cloud'],
                'color': '#74B9FF'
            },
            'Politik & Ranah Publik': {
                'words': ['politik', 'partai', 'pemilu', 'pilkada', 'calon', 'kandidat', 'kampanye', 'suara', 'voting', 'demokrasi', 'rakyat', 'masyarakat', 'warga', 'negara', 'kpu', 'bawaslu', 'dpr', 'dprd', 'mpr', 'dpd', 'anggota', 'wakil', 'fraksi', 'koalisi', 'oposisi', 'debat', 'sidang', 'rapat', 'musyawarah', 'mufakat'],
                'color': '#FD79A8'
            },
            'Sosial & Masyarakat': {
                'words': ['sosial', 'masyarakat', 'komunitas', 'kelompok', 'organisasi', 'lsm', 'ngo', 'relawan', 'volunteer', 'gotong', 'royong', 'kerja', 'sama', 'tolong', 'menolong', 'bantuan', 'donasi', 'sedekah', 'zakat', 'infaq', 'shadaqah', 'kegiatan', 'acara', 'event', 'festival', 'budaya', 'adat', 'tradisi', 'agama', 'islam'],
                'color': '#A29BFE'
            },
            'Ekonomi & Keuangan': {
                'words': ['ekonomi', 'keuangan', 'uang', 'rupiah', 'dollar', 'bank', 'kredit', 'pinjaman', 'investasi', 'saham', 'obligasi', 'reksadana', 'asuransi', 'bisnis', 'usaha', 'dagang', 'jual', 'beli', 'harga', 'mahal', 'murah', 'diskon', 'promo', 'untung', 'rugi', 'modal', 'keuntungan', 'pendapatan', 'gaji', 'upah'],
                'color': '#FDCB6E'
            }
        }
        
    def categorize_word(self, word):
        """Categorize a word based on semantic categories"""
        word_lower = word.lower()
        for category, info in self.word_categories.items():
            if word_lower in info['words'] or any(w in word_lower for w in info['words']):
                return category, info['color']
        return 'Lainnya', '#95A5A6'  # Default category
        
    def create_umap_visualization(self, query_tokens, top_document_tokens, max_words=200):
        """
        Create UMAP visualization of Word2Vec vectors with semantic categorization
        """
        try:
            # Collect words for visualization
            words_to_visualize = set()
            
            # Add query tokens
            for token in query_tokens[:20]:  # More query tokens
                if token in self.word2vec_model.model.wv:
                    words_to_visualize.add(token)
            
            # Add top document tokens
            for doc_tokens in top_document_tokens[:10]:  # More documents
                for token in doc_tokens[:20]:  # More tokens per document
                    if token in self.word2vec_model.model.wv:
                        words_to_visualize.add(token)
                        if len(words_to_visualize) >= max_words:
                            break
                if len(words_to_visualize) >= max_words:
                    break
            
            # Add words from predefined categories to ensure good representation
            for category, info in self.word_categories.items():
                for word in info['words'][:10]:  # Add some words from each category
                    if word in self.word2vec_model.model.wv:
                        words_to_visualize.add(word)
                        if len(words_to_visualize) >= max_words:
                            break
                if len(words_to_visualize) >= max_words:
                    break
            
            words_list = list(words_to_visualize)
            
            if len(words_list) < 10:
                logging.warning("Not enough words for visualization")
                return None
            
            # Get word vectors
            word_vectors = []
            final_words = []
            
            for word in words_list:
                if word in self.word2vec_model.model.wv:
                    word_vectors.append(self.word2vec_model.model.wv[word])
                    final_words.append(word)
            
            if len(word_vectors) < 10:
                logging.warning("Not enough valid word vectors for visualization")
                return None
            
            word_vectors = np.array(word_vectors)
            
            # Apply UMAP for dimensionality reduction
            if len(word_vectors) >= 15:
                reducer = umap.UMAP(
                    n_neighbors=min(15, len(word_vectors) - 1),
                    min_dist=0.1,
                    n_components=2,
                    random_state=42,
                    spread=1.0
                )
                embedding = reducer.fit_transform(word_vectors)
            else:
                pca = PCA(n_components=2, random_state=42)
                embedding = pca.fit_transform(word_vectors)
            
            # Create visualization with larger figure
            plt.figure(figsize=(16, 12))
            
            # Categorize words and prepare colors
            word_categories = {}
            colors = []
            category_counts = {}
            
            for word in final_words:
                category, color = self.categorize_word(word)
                word_categories[word] = category
                colors.append(color)
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Create scatter plot
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=colors, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            
            # Add word labels with better positioning
            for i, word in enumerate(final_words):
                plt.annotate(word, (embedding[i, 0], embedding[i, 1]), 
                           xytext=(2, 2), textcoords='offset points', 
                           fontsize=8, alpha=0.9, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Create legend
            legend_elements = []
            for category, info in self.word_categories.items():
                if category in category_counts:
                    legend_elements.append(
                        mpatches.Patch(color=info['color'], label=f"{category} ({category_counts[category]})")
                    )
            
            # Add "Lainnya" if there are uncategorized words
            if 'Lainnya' in category_counts:
                legend_elements.append(
                    mpatches.Patch(color='#95A5A6', label=f"Lainnya ({category_counts['Lainnya']})")
                )
            
            # Position legend outside the plot
            plt.legend(handles=legend_elements, title='Kategori', 
                      bbox_to_anchor=(1.05, 1), loc='upper left', 
                      fontsize=10, title_fontsize=12)
            
            # Styling
            plt.title('UMAP Visualization of Word2Vec Vectors', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('UMAP Component 1', fontsize=12)
            plt.ylabel('UMAP Component 2', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Set background color
            plt.gca().set_facecolor('#FAFAFA')
            
            # Adjust layout to prevent legend cutoff
            plt.tight_layout()
            
            # Save the plot
            save_file = os.path.join(self.save_path, 'Visualition_Word2vec.png')
            plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            return save_file
            
        except Exception as e:
            logging.error(f"Error creating UMAP visualization: {e}")
            return None
    
    def get_top_document_tokens(self, results, doc_texts):
        """Get tokens from top documents for visualization"""
        top_document_tokens = []
        
        for result in results[:10]:  # Top 10 documents
            doc_id = result['id']
            if doc_id in doc_texts:
                # Get document text and tokenize
                doc_text = doc_texts[doc_id]
                # Simple tokenization - split by spaces and clean
                doc_tokens = []
                words = doc_text.lower().split()
                for word in words[:50]:  # First 50 words
                    # Clean word (remove punctuation)
                    clean_word = ''.join(c for c in word if c.isalpha())
                    if len(clean_word) > 2:  # Only words longer than 2 characters
                        doc_tokens.append(clean_word)
                
                if doc_tokens:
                    top_document_tokens.append(doc_tokens)
        
        return top_document_tokens
