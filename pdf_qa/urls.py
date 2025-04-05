from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('qa/<int:document_id>/', views.question_answering, name='qa'),
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('delete/<int:document_id>/', views.delete_pdf, name='delete_pdf'),
]
