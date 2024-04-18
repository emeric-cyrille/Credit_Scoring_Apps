from django.shortcuts import render
from django.http import JsonResponse
from django.urls import reverse


# Create your views here.

def index(request):

    return render(request, 'index.html')


def get_columns(request, dataset_id):
    columns = Column.objects.filter(dataset_id=dataset_id).values('id', 'name')
    return JsonResponse({'columns': list(columns)})


from django.shortcuts import render
from django.http import HttpResponseRedirect
from .models import Dataset, Model, Column, SelectedColumn

def create_model(request):
    datasets = Dataset.objects.all()  # Récupérer tous les datasets
    models = Model.objects.all()  # Récupérer tous les modèles disponibles
    algorithm_choices = Model.STATUS_CHOICES  # Récupérer les choix d'algorithmes depuis le modèle
    return render(request, 'create_model.html', {'datasets': datasets, 'models': models, 'algorithm_choices': algorithm_choices})

def save_model(request):
    if request.method == 'POST':
        # Récupérer les données soumises par le formulaire
        dataset_ids = request.POST.getlist('datasets')  # Récupérer les IDs des datasets sélectionnés
        columns_ids = request.POST.getlist('columns')  # Récupérer les IDs des colonnes sélectionnées
        model_id = request.POST.get('model')  # Récupérer l'ID du modèle sélectionné

        # Enregistrer les colonnes sélectionnées pour chaque dataset dans la base de données
        for dataset_id in dataset_ids:
            dataset = Dataset.objects.get(pk=dataset_id)
            for column_id in columns_ids:
                column = Column.objects.get(pk=column_id)
                selected_column = SelectedColumn(dataset=dataset, column=column)
                selected_column.save()

        # Enregistrer le modèle sélectionné dans la base de données (si nécessaire)
        # Rediriger vers une autre page après la sauvegarde
        return HttpResponseRedirect('/success/')  # Exemple : Redirection vers une page de succès

    # Rediriger vers la page de création de modèle en cas de requête GET
    return HttpResponseRedirect('/create_model/')


import os
import pickle
from django.shortcuts import render, redirect
from .models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def train_model(request):
    if request.method == 'POST':
        try:
            # Récupérer les données de la requête POST
            algorithm = request.POST.get('algorithm')
            dataset_path = request.POST.get('dataset')
            selected_columns = request.POST.getlist('columns[]')
            model_name = request.POST.get('model_name')  # Récupérer le nom du modèle

            # Charger le dataset et ne garder que les colonnes sélectionnées
            dataset = pd.read_csv(dataset_path)
            dataset = dataset[selected_columns]

            # Séparer les colonnes en entrées et colonne cible
            input_columns = [col for col in selected_columns if col != 'target']
            target_column = 'target'

            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(dataset[input_columns], dataset[target_column],
                                                                test_size=0.2, random_state=42)

            # Créer et entraîner le modèle en fonction de l'algorithme choisi
            if algorithm == 'logistic_regression':
                model = LogisticRegression()
            elif algorithm == 'random_forest':
                model = RandomForestClassifier()
            elif algorithm == 'decision_tree':
                model = DecisionTreeClassifier()

            model.fit(X_train, y_train)

            # Sauvegarder le modèle entraîné dans un fichier pkl
            model_data_dir = os.path.join('models', 'model_data')
            os.makedirs(model_data_dir, exist_ok=True)
            model_data_path = os.path.join(model_data_dir, '{}.pkl'.format(model_name))
            with open(model_data_path, 'wb') as f:
                pickle.dump(model, f)

            # Rediriger vers la vue de sauvegarde du modèle en passant les informations nécessaires
            return redirect(reverse('save_model', kwargs={'model_name': model_name, 'algorithm': algorithm,
                                                          'model_data': model_data_path, 'dataset': dataset_path}))
        except Exception as e:
            # En cas d'erreur, retourner une réponse JSON avec un message d'erreur
            return render(request, 'error.html', {'message': "Une erreur s'est produite lors de l'entraînement du modèle : {}".format(str(e))})

    else:
        # Retourner une réponse JSON indiquant que la méthode HTTP n'est pas autorisée
        return render(request, 'error.html', {'message': "Méthode HTTP non autorisée"})

from django.http import JsonResponse

def save_model(request, model_name, algorithm, model_data, dataset):
    try:
        # Créer une nouvelle instance du modèle avec les informations d'entraînement
        trained_model = Model(name=model_name, algorithm=algorithm, model_data=model_data, dataset=dataset)
        trained_model.save()

        # Retourner une réponse JSON avec un message de succès
        return JsonResponse({'success': True, 'message': "Le modèle '{}' a été entraîné avec succès et enregistré.".format(model_name)})

    except Exception as e:
        # En cas d'erreur, retourner une réponse JSON avec un message d'erreur
        return JsonResponse({'success': False, 'message': "Une erreur s'est produite lors de l'enregistrement du modèle : {}".format(str(e))})
