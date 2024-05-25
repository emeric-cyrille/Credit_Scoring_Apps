from django.db import models

class Dataset(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(default='Mettre la description ici')
    data = models.FileField(upload_to='datasets/')
    apercu = models.ImageField(upload_to='dataset_previews/', blank=True, null=True)
    #target_column = models.CharField(max_length=100)  # Ajoutez ce champ si nécessaire

    def __str__(self):
        return self.name

class Model(models.Model):
    name = models.CharField(max_length=100)
    ALGO_CHOICES = (
        ('randomforest', 'RamdomForest'),
        ('logistic_regression', 'Regression Logistique'),
        ('decision_tree', 'Arbre de décision'),
        ('xgboost', 'XGBOOST'),
        ('sfs', 'SFS'),
        ('lasso', 'LASSO'),
        ('relief', 'Relief')
    )
    algorithm = models.CharField(max_length=100, choices=ALGO_CHOICES, default='Regression Logistique')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model_data = models.FileField(upload_to='static/models/', blank=True, null=True)
    STATUS_CHOICES = (
        ('trained', 'Entrainé'),
        ('untrained', 'Non entrainé')
    )
    status = models.CharField(max_length=100, choices=STATUS_CHOICES, default='untrained')
    accuracy = models.FloatField(default=0)
    def __str__(self):
        return self.name

class Column(models.Model):
    #id = models.AutoField(primary_key=True)
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
