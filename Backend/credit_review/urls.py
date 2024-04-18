from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='home'),

    path('create_model/', views.create_model, name='create_model'),
    path('save_model/<str:model_name>/<str:algorithm>/<str:model_data>/<str:dataset>/', views.save_model, name='save_model'),
    path('get_columns/<int:dataset_id>/', views.get_columns, name='get_columns'),
    path('train_model/', views.train_model, name='train_model'),
]