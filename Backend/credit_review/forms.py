from django import forms
from .models import Dataset

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'data', 'apercu']  # Les champs que vous souhaitez inclure dans le formulaire
