from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='home'),
    path('create_model/', views.save_model, name='create_model'),
    path('get_columns/<int:dataset_id>/', views.get_columns, name='get_columns'),
    path('list_models/', views.list_models, name='list_models'),
    path('model/<int:model_id>/', views.model_details, name='model_details'),
    path('train_model/', views.train_model, name='train_model'),
    path('predict/', views.predict, name='predict'),
    path('get_selected_columns/<int:model_id>/', views.get_selected_columns, name='get_selected_columns'),
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('upload/success/', views.upload_success, name='upload_success'),


]