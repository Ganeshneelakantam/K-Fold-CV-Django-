from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Render the index page
    path('documentation/', views.documentation, name='documentation'),
    path('upload/', views.upload, name='upload'),  # Handle file uploads
]
