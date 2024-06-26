
from django.db import models

class Dataset(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(default='Mettre la description ici')
    data = models.FileField(upload_to='datasets/')
    apercu = models.ImageField(upload_to='dataset_previews/', blank=True, null=True)

    def __str__(self):
        return self.name

class Model(models.Model):
    name = models.CharField(max_length=100)
    ALGO_CHOICES = (
        ('randomforest', 'RamdomForest'),
        ('logistic_regression', 'Regression Logistique'),
        ('decision_tree', 'Arbre de décision'),
        ('xgboost', 'XGBOOST'),
        ('svm', 'SVM'),
        ('igboost', 'IGBOOST')
    )
    algorithm = models.CharField(max_length=100, choices=ALGO_CHOICES, default='Regression Logistique')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model_data = models.FileField(upload_to='models/', blank=True, null=True)
    STATUS_CHOICES = (
        ('trained', 'Entrainé'),
        ('untrained', 'Non entrainé')
    )
    status = models.CharField(max_length=100, choices=STATUS_CHOICES, default='Non entrainé')
    accuracy = models.FloatField(default=0)
    def __str__(self):
        return self.name

class Column(models.Model):
    name = models.CharField(max_length=100)
    STATUS_CHOICES = (
        ('input', 'Input'),
        ('target', 'Target'),
    )
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Input')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class SelectedColumn(models.Model):
    column = models.ForeignKey(Column, on_delete=models.CASCADE)
    model = models.ForeignKey(Model, on_delete=models.CASCADE)
