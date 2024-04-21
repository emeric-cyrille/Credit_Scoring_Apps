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

from django.shortcuts import render, redirect
from .models import Model, SelectedColumn
from django.contrib import messages


def save_model(request):
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        dataset_id = request.POST.get('dataset')
        algorithm = request.POST.get('algorithm')
        selected_columns = request.POST.getlist('columns')

        # Enregistrer le modèle
        model = Model.objects.create(name=model_name, algorithm=algorithm, dataset_id=dataset_id)

        # Enregistrer les colonnes sélectionnées pour ce modèle
        for column_id in selected_columns:
            column = SelectedColumn.objects.create(column_id=column_id, model=model)

        messages.success(request, 'Le modèle a été enregistré avec succès.')
        return redirect('home')  # Rediriger vers la page d'accueil ou une autre vue après l'enregistrement du modèle

    return render(request, 'create_model.html', {'datasets': Dataset.objects.all()})



def list_models(request):
    trained_datasets = Model.objects.exclude(status='untrained').distinct()
    untrained_datasets = Model.objects.filter(status='untrained').distinct()
    context = {
        'trained_datasets': trained_datasets,
        'untrained_datasets': untrained_datasets,
    }
    return render(request, 'list_model.html', context)



from django.shortcuts import render, get_object_or_404
from .models import Model

def model_details(request, model_id):
    model = get_object_or_404(Model, pk=model_id)
    return render(request, 'detail_model.html', {'model': model})







from django.shortcuts import render, redirect
from django.urls import reverse
from .models import Model, SelectedColumn, Column
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_model(request):
    if request.method == 'POST':
        try:
            # Récupérer l'ID du modèle à entraîner depuis la requête POST
            model_id = request.POST.get('model_id')

            # Récupérer les données du modèle existant
            model = Model.objects.get(pk=model_id)

            # Charger les informations du modèle depuis la base de données
            algorithm = model.algorithm
            dataset_path = model.dataset.data.path

            # Récupérer les colonnes sélectionnées associées au modèle
            selected_columns = SelectedColumn.objects.filter(model=model)
            column_names = [sc.column.name for sc in selected_columns]

            # Récupérer les noms de toutes les colonnes disponibles
            all_columns = Column.objects.filter(dataset=model.dataset)
            all_column_names = [col.name for col in all_columns]

            # Vérifier si toutes les colonnes sélectionnées sont disponibles dans le dataset
            for col_name in column_names:
                if col_name not in all_column_names:
                    raise ValueError(f"La colonne '{col_name}' n'est pas disponible dans le dataset.")

            # Déterminer la colonne cible (target) en fonction de l'attribut 'status' de la colonne
            target_column = [col.name for col in all_columns if col.status == 'target'][0]
            input_columns = [col for col in column_names if col != target_column]

            # Charger le dataset et ne garder que les colonnes sélectionnées
            dataset = pd.read_csv(dataset_path)
            dataset = dataset[column_names]

            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(dataset[input_columns], dataset[target_column],
                                                                test_size=0.2, random_state=42)


            # Créer et entraîner le modèle en fonction de l'algorithme choisi
            if algorithm == 'logistic_regression':
                model_algorithm = LogisticRegression()
            elif algorithm == 'random_forest':
                model_algorithm = RandomForestClassifier()
            elif algorithm == 'decision_tree':
                model_algorithm = DecisionTreeClassifier()

            model_algorithm.fit(X_train, y_train)

            # Faire des prédictions sur l'ensemble de test pour calculer l'accuracy
            y_pred = model_algorithm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Sauvegarder le modèle entraîné dans un fichier pkl
            model_data_dir = os.path.join('models')
            os.makedirs(model_data_dir, exist_ok=True)
            model_data_path = os.path.join(model_data_dir, '{}.pkl'.format(model.name))
            joblib.dump(model_algorithm, model_data_path)

            # Mettre à jour les informations du modèle dans la base de données
            model.model_data = model_data_path
            model.status = 'trained'
            model.accuracy = round(accuracy * 100, 2)
            model.save()

            # Rediriger vers la vue de détails du modèle avec son ID
            return redirect(reverse('model_details', kwargs={'model_id': model_id}))
        except Exception as e:
            # En cas d'erreur, retourner une réponse avec un message d'erreur
            return render(request, 'error.html', {'message': "Une erreur s'est produite lors de l'entraînement du modèle : {}".format(str(e))})
    else:
        # Retourner une réponse indiquant que la méthode HTTP n'est pas autorisée
        return render(request, 'error.html', {'message': "Méthode HTTP non autorisée"})


def predict(request):
    if request.method == 'POST':
        try:
            # Récupérer l'ID du modèle sélectionné dans le formulaire
            model_id = request.POST.get('model_id')
            model = Model.objects.get(pk=model_id)

            # Récupérer les colonnes sélectionnées associées au modèle
            selected_columns = SelectedColumn.objects.filter(model=model)

            # Récupérer les valeurs des colonnes saisies par l'utilisateur
            input_data = {}
            for column in selected_columns:
                input_data[column.column.name] = request.POST.get(column.column.name)

            # Charger le modèle depuis le fichier stocké
            model_path = model.model_data.path
            loaded_model = joblib.load(model_path)

            # Effectuer la prédiction sur les données saisies par l'utilisateur
            prediction = loaded_model.predict([list(input_data.values())])

            context = {
                'model': model,
                'input_data': input_data,
                'prediction': prediction[0]  # Assumant que le modèle renvoie une seule prédiction
            }
            return render(request, 'prediction_result.html', context)
        except Exception as e:
            # En cas d'erreur, retourner une réponse avec un message d'erreur
            return render(request, 'error.html',
                          {'message': "Une erreur s'est produite lors de la prédiction : {}".format(str(e))})
    else:
        # Récupérer tous les modèles entraînés
        trained_models = Model.objects.filter(status='trained')
        context = {
            'trained_models': trained_models
        }
        return render(request, 'predict.html', context)