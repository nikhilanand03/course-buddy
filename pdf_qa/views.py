from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from .models import PDFDocument
from .utils import process_pdf, get_answer
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import os

def index(request):
    documents = PDFDocument.objects.all().order_by('-uploaded_at')
    return render(request, 'pdf_qa/index.html', {'documents': documents})

@csrf_exempt
def upload_pdf(request):
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        pdf_file = request.FILES['pdf_file']
        title = pdf_file.name
        
        # Save the document to get an ID
        document = PDFDocument.objects.create(
            title=title,
            file=pdf_file
        )
        
        # Process the PDF and create vector store
        try:
            vector_store_path = process_pdf(document.file.path, document.id)
            document.vector_store_path = vector_store_path
            document.save()
            messages.success(request, 'PDF uploaded and processed successfully!')
        except Exception as e:
            document.delete()  # Clean up if processing fails
            messages.error(request, f'Error processing PDF: {str(e)}')
            
        return redirect('index')
    
    return redirect('index')

def question_answering(request, document_id):
    document = get_object_or_404(PDFDocument, id=document_id)
    
    if request.method == 'POST' and 'question' in request.POST:
        question = request.POST['question']
        
        try:
            result = get_answer(question, document.vector_store_path)
            
            context = {
                'document': document,
                'question': question,
                'answer': result['answer'],
                'source_pages': result['source_pages']
            }
            return render(request, 'pdf_qa/qa.html', context)
        except Exception as e:
            messages.error(request, f'Error generating answer: {str(e)}')
    
    return render(request, 'pdf_qa/qa.html', {'document': document})

def delete_pdf(request, document_id):
    document = get_object_or_404(PDFDocument, id=document_id)
    document.delete()
    messages.success(request, 'PDF deleted successfully!')
    return redirect('index')
