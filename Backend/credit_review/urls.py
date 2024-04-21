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

]