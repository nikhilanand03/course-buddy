from django.db import models

# Create your models here.
from django.db import models
import os

class PDFDocument(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='pdfs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    vector_store_path = models.CharField(max_length=255, blank=True, null=True)
    
    def __str__(self):
        return self.title
    
    def delete(self, *args, **kwargs):
        # Delete the file from storage
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        
        # Delete vector store if it exists
        if self.vector_store_path and os.path.isdir(self.vector_store_path):
            import shutil
            shutil.rmtree(self.vector_store_path)
            
        super().delete(*args, **kwargs)
