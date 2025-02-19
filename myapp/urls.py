# myapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_dataset, name='upload_dataset'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('user_journey/', views.user_journey, name='user_journey'),
]