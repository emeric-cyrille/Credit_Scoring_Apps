from django.contrib import admin

# Register your models here.
# admin.py

from django.contrib import admin
from .models import Dataset, Model, Column, SelectedColumn

admin.site.register(Dataset)
admin.site.register(Model)
admin.site.register(Column)
admin.site.register(SelectedColumn)
