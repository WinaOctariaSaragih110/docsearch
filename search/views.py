import os
import shutil
import datetime
import json
import time
import hashlib
import sys
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from .models import Document, Result
from .search_engine import get_search_engine
from .ocr_processor import OCRProcessor
import logging
import threading

# Set maximum upload size to 200 KB
MAX_UPLOAD_SIZE = 200 * 1024  # 200 KB in bytes

def index(request):
    """Main page view - show only testing documents"""
    engine = get_search_engine()
    testing_documents = engine.data_splitter.get_testing_documents().order_by('id')
    
    # Get split information for display
    split_info = engine.data_splitter.get_split_info()
    
    return render(request, 'index.html', {
        'documents': testing_documents,
        'split_info': split_info
    })

@csrf_exempt
def search_documents(request):
    """Handle document search - search only in testing documents"""
    if request.method == 'POST' and request.FILES.get('image'):
        # Start timing from OCR process
        search_start_time = time.time()
        
        # Get uploaded image
        image_file = request.FILES['image']
        
        # Process OCR
        ocr_processor = OCRProcessor()
        ocr_text = ocr_processor.process_image(image_file)
        
        # Clear previous results to ensure fresh results for each search
        Result.objects.all().delete()
        
        # Check if the image contains text
        if ocr_text == "GAMBAR TIDAK BERISI TEKS":
            # Display notification and stop processing
            messages.error(request, 'GAMBAR TIDAK BERISI TEKS')
            # Return to index without any search results or OCR text
            return redirect('index')
        
        # If we get here, the image contains text, so continue with search
        engine = get_search_engine()
        results, _ = engine.search(ocr_text, search_start_time=search_start_time)
        
        # Get top 10 results
        top_results = results[:10]
        
        # Get testing documents for display
        testing_documents = engine.data_splitter.get_testing_documents().order_by('id')
        split_info = engine.data_splitter.get_split_info()
        
        # Run evaluation automatically after search completes
        def run_evaluation_after_response():
            try:
                # Set matplotlib backend untuk thread ini
                import matplotlib
                matplotlib.use('Agg')
                
                # Wait a moment to ensure response is fully sent
                time.sleep(1.0)
                
                # Force flush stdout to ensure output appears
                sys.stdout.flush()
                sys.stderr.flush()
                
                # Print separator to make output visible
                print("\n" + "="*80)
                print("EVALUASI SISTEM PENCARIAN DOKUMEN")
                print("="*80)
                sys.stdout.flush()
                
                # Run the evaluation automatically
                engine.run_evaluation_async()
                
                # Force flush again
                sys.stdout.flush()
                sys.stderr.flush()
                
                print("="*80)
                print("EVALUASI SELESAI")
                print("="*80 + "\n")
                sys.stdout.flush()
                
            except Exception as e:
                print(f"\nERROR in evaluation thread: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
            finally:
                # Clean up matplotlib resources
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                except:
                    pass
        
        # Start evaluation in a separate non-daemon thread
        eval_thread = threading.Thread(target=run_evaluation_after_response)
        eval_thread.daemon = False  # Non-daemon thread
        eval_thread.start()
        
        return render(request, 'index.html', {
            'documents': testing_documents,
            'results': top_results,
            'ocr_text': ocr_text,
            'search_performed': True,
            'split_info': split_info
        })

    return redirect('index')

@csrf_exempt
def add_document(request):
    """Add a new document - automatically goes to testing set"""
    if request.method == 'POST' and request.FILES.get('document'):
        document_file = request.FILES['document']
        
        # Check if file is PDF
        if not document_file.name.lower().endswith('.pdf'):
            messages.error(request, 'Hanya file PDF yang diperbolehkan')
            return redirect('index')
        
        # Check file size (max 200KB)
        if document_file.size > MAX_UPLOAD_SIZE:
            messages.error(request, 'File Lebih dari 200 kb')
            return redirect('index')
        
        # Create a temporary file to check for duplicate content
        temp_dir = os.path.join(settings.BASE_DIR, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, document_file.name)
        
        with open(temp_path, 'wb+') as destination:
            for chunk in document_file.chunks():
                destination.write(chunk)
        
        # Calculate file hash for content comparison
        file_hash = calculate_file_hash(temp_path)
        
        # Check for duplicate title
        document_name = os.path.splitext(document_file.name)[0]
        if Document.objects.filter(document_name=document_name).exists():
            # Clean up temporary file
            os.remove(temp_path)
            messages.error(request, 'Dokumen sudah ada didalam database')
            return redirect('index')
        
        # Check for duplicate content by comparing file hash
        duplicate_content = False
        for doc in Document.objects.all():
            doc_path = doc.document_path
            if not os.path.isabs(doc_path):
                doc_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(doc_path))
            
            if os.path.exists(doc_path):
                existing_hash = calculate_file_hash(doc_path)
                if existing_hash == file_hash:
                    duplicate_content = True
                    break
        
        if duplicate_content:
            # Clean up temporary file
            os.remove(temp_path)
            messages.error(request, 'Dokumen sudah ada didalam database')
            return redirect('index')
        
        # Save file to disk
        pdf_storage_path = settings.PDF_STORAGE_PATH
        os.makedirs(pdf_storage_path, exist_ok=True)
        
        # Create a safe filename
        filename = document_file.name
        file_path = os.path.join(pdf_storage_path, filename)
        
        # Move the temporary file to the final location
        shutil.move(temp_path, file_path)
        
        # Create relative path for database
        relative_path = os.path.join('static/pdfdocuments', filename)
        
        # Add to database with transaction to ensure ID is updated correctly
        with transaction.atomic():
            # Get the highest ID
            max_id = Document.objects.all().order_by('-id').first()
            new_id = 1 if max_id is None else max_id.id + 1
            
            # Create new document record - automatically goes to testing set (is_training=False)
            document = Document(
                id=new_id,
                document_name=document_name,
                document_uploaddate=datetime.date.today(),
                document_path=relative_path,
                is_training=False  # New documents go to testing set
            )
            document.save()
            
            # Reorder IDs to ensure they are sequential
            reorder_document_ids()
        
        # Add new document to search engine
        engine = get_search_engine()
        engine.add_new_document(document.id)

        # Log the change
        print(f"\n[INFO] Dokumen baru '{document_name}' ditambahkan ke testing set")
        print(f"[INFO] ID dokumen: {document.id}")
        print(f"[INFO] Transformer akan direbuild pada pencarian berikutnya")
        
        messages.success(request, 'Dokumen berhasil ditambahkan ke testing set')
        return redirect('index')

    return redirect('index')

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file for content comparison"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

@csrf_exempt
def delete_document(request, document_id):
    """Delete a document"""
    try:
        with transaction.atomic():
            # Get the document
            document = Document.objects.get(id=document_id)
            
            # Delete the file
            file_path = document.document_path
            if not os.path.isabs(file_path):
                file_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(file_path))
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from search engine index
            engine = get_search_engine()
            engine.remove_document_from_index(document_id)

            # Clear document processor cache
            engine.document_processor.clear_document_cache(file_path)

            # Delete from database
            document.delete()

            # Reorder IDs to ensure they are sequential
            reorder_document_ids()

            print(f"\n[INFO] Dokumen ID {document_id} berhasil dihapus dari testing set")
            print("[INFO] Index pencarian telah diperbarui")
            print("[INFO] Transformer akan direbuild pada pencarian berikutnya")
        
        messages.success(request, 'Dokumen berhasil dihapus')
        return redirect('index')
    except Document.DoesNotExist:
        messages.error(request, 'Dokumen tidak ditemukan')
        return redirect('index')
    except Exception as e:
        messages.error(request, f'Error: {str(e)}')
        return redirect('index')

def download_document(request, document_id):
    """Download a document"""
    try:
        document = Document.objects.get(id=document_id)
        file_path = document.document_path
        
        if not os.path.isabs(file_path):
            file_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(file_path))
        
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))
        else:
            messages.error(request, 'File tidak ditemukan')
            return redirect('index')
    except Document.DoesNotExist:
        messages.error(request, 'Dokumen tidak ditemukan')
        return redirect('index')
    except Exception as e:
        messages.error(request, str(e))
        return redirect('index')

def reorder_document_ids():
    """Reorder document IDs to ensure they are sequential"""
    documents = Document.objects.all().order_by('id')
    
    # Update IDs to be sequential
    for i, doc in enumerate(documents, start=1):
        if doc.id != i:
            Document.objects.filter(id=doc.id).update(id=i)
