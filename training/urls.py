from django.urls import path
from . import views

app_name = 'training'

urlpatterns = [
    path('', views.get_training_data, name='get_training_data'),
    path('delete/', views.delete_training_data, name='delete_training_data'),
]