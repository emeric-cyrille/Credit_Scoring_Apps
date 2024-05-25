from django.urls import reverse
from django.shortcuts import render, redirect
from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split

from .forms import DatasetForm


# Create your views here.

def index(request):
    return render(request, 'index.html')


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
    #trained_datasets = Model.objects.filter(status='trained').distinct()
    trained_datasets = Model.objects.filter(status='trained').distinct()
    untrained_datasets = Model.objects.filter(status='untrained').distinct()
    context = {
        'trained_datasets': trained_datasets,
        'untrained_datasets': untrained_datasets,
    }
    return render(request, 'list_model.html', context)



from django.shortcuts import render, get_object_or_404

def model_details(request, model_id):
    model = get_object_or_404(Model, pk=model_id)
    return render(request, 'detail_model.html', {'model': model})





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
            print(algorithm)
            print(dataset_path)
            # Récupérer les colonnes sélectionnées associées au modèle
            # Récupérer les colonnes sélectionnées associées au modèle
            selected_columns = SelectedColumn.objects.filter(model=model)
            column_names = [sc.column.name for sc in selected_columns]
            print(column_names)

            # Récupérer les noms de toutes les colonnes disponibles
            all_columns = Column.objects.filter(dataset=model.dataset)
            all_column_names = [col.name for col in all_columns]
            print(all_column_names)

            # Vérifier si toutes les colonnes sélectionnées sont disponibles dans le dataset
            for col_name in column_names:
                if col_name not in all_column_names:
                    raise ValueError(f"La colonne '{col_name}' n'est pas disponible dans le dataset.")

            # Déterminer la colonne cible (target) en fonction de l'attribut 'status' de la colonne
            target_column = [col.name for col in all_columns if col.status == 'target'][0]
            input_columns = [col for col in column_names if col != target_column]



            print(target_column)
            print(input_columns)
            # Charger le dataset et ne garder que les colonnes sélectionnées
            dataset = pd.read_csv(dataset_path)
            print(dataset)
            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(dataset[input_columns], dataset[str(target_column)],
                                                                test_size=0.2, random_state=42)


            # Créer et entraîner le modèle en fonction de l'algorithme choisi
            if algorithm == 'logistic_regression':
                model_algorithm = LogisticRegression()
            elif algorithm == 'randomforest':
                model_algorithm = RandomForestClassifier()
            elif algorithm == 'decision_tree':
                model_algorithm = DecisionTreeClassifier()

            model_algorithm.fit(X_train, y_train)

            # Faire des prédictions sur l'ensemble de test pour calculer l'accuracy
            y_pred = model_algorithm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Sauvegarder le modèle entraîné dans un fichier pkl
            model_data_dir = os.path.join('static/models')
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



def get_selected_columns(request, model_id):
    try:
        # Récupérer toutes les colonnes sélectionnées avec le statut "input" pour le modèle donné
        selected_columns = SelectedColumn.objects.filter(model_id=model_id, column__status='input').values_list('column__name', flat=True)

        # Retourner les colonnes sélectionnées sous forme de liste JSON
        return JsonResponse({'selected_columns': list(selected_columns)})
    except Exception as e:
        # En cas d'erreur, retourner une réponse avec un message d'erreur
        return JsonResponse({'error': str(e)}, status=500)


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

from django.shortcuts import render, redirect
from django.urls import reverse
from .models import Dataset, Column
import csv
from django.http import JsonResponse
from .models import Column
from io import TextIOWrapper
import json

def upload_dataset(request):
    if request.method == 'POST':
        # Récupérer les données du formulaire
        dataset_name = request.POST.get('dataset_name')
        description = request.POST.get('description')
        target_column = request.POST.get('target_column')
        data_file = request.FILES.get('data')
        apercu = request.FILES.get('apercu')
        columns = json.loads(request.POST.get('columns'))


        print(target_column)
        # Enregistrer le fichier de données dans la base de données
        dataset = Dataset.objects.create(name=dataset_name, description=description, data=data_file, apercu=apercu)


        print(columns)
        # Enregistrer les colonnes dans la base de données
        for column_name in columns:
            status = 'target' if column_name == target_column else 'input'
            column = Column.objects.create(dataset=dataset, name=column_name, status=status)
            print(column)

        # Rediriger vers une page de confirmation
        return redirect(reverse('upload_success'))

    return render(request, 'upload_dataset.html')


def upload_success(request):
    return render(request, 'upload_success.html')


from django.shortcuts import render, redirect
from .models import Dataset
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .models import Dataset, Column


from django.shortcuts import render, redirect
from django.urls import reverse
from .models import Dataset, Column
from .feature_selection import run_feature_selection

def feature_selection(request):
    datasets = Dataset.objects.all()  # Récupérer tous les datasets de la base de données
    datasets_info = []

    for dataset in datasets:
        columns = Column.objects.filter(dataset=dataset)
        non_target_columns = columns.exclude(status='target').count()
        datasets_info.append({'id': dataset.id, 'name': dataset.name, 'num_columns': non_target_columns})

    if request.method == 'POST':
        dataset_id = request.POST.get('dataset')
        algorithm = request.POST.get('algorithm')
        k = int(request.POST.get('k', 10))  # Valeur par défaut de 10 si k n'est pas fourni
        num_features = int(request.POST.get('num_features', 10))  # Valeur par défaut de 10 si num_features n'est pas fourni

        # Lancer l'algorithme de sélection d'attributs
        selected_features = run_feature_selection(dataset_id, algorithm, k, num_features)

        # Rediriger vers la page des résultats avec les colonnes significatives
        return redirect(reverse('feature_selection_results') + f'?dataset={dataset_id}&algorithm={algorithm}&k={k}&num_features={num_features}')

    return render(request, 'feature_selection.html', {'datasets': datasets, 'datasets_info': datasets_info})


from .feature_selection import run_feature_selection
from django.shortcuts import render, redirect
from django.urls import reverse
from .models import Dataset, Model

"""
def feature_selection_results(request):
    if request.method == 'POST':
        selected_columns = request.POST.getlist('selected_columns')
        dataset_id = request.POST.get('dataset_id')
        model_name = request.POST.get('model_name')

        # Rediriger vers la vue de création de modèle avec les colonnes sélectionnées
        query_params = {
            'selected_columns': selected_columns,
            'dataset': dataset_id,
            'model_name': model_name,
            'action': 'feature_selection'
        }
        return redirect(reverse('create_model_algo') + '?' + urlencode(query_params))


    dataset_id = request.GET.get('dataset')
    algorithm = request.GET.get('algorithm')
    k = int(request.GET.get('k', 10))
    num_features = int(request.GET.get('num_features', 10))

    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        return render(request, 'feature_selection_results.html', {'error_message': 'Dataset not found.'})

    selected_features = run_feature_selection(dataset_id, algorithm, k, num_features)

    # Récupérer le nom du modèle précédemment choisi, si disponible
    model_name = request.session.get('model_name', 'Unknown Model')


    # Créer l'URL de retour avec les informations nécessaires
    return_url = reverse('create_model_algo') + f'?dataset={dataset_id}&model_name={model_name}'

    return render(request, 'feature_selection_results.html', {
        'dataset': dataset,
        'algorithm': algorithm,
        'selected_features': selected_features,
        'num_features': num_features,
        'return_url': return_url
    })
"""
def feature_selection_results(request):
    if request.method == 'POST':
        selected_columns = request.POST.getlist('selected_columns')
        dataset_id = request.POST.get('dataset_id')
        model_name = request.POST.get('model_name')

        # Rediriger vers la vue de création de modèle avec les colonnes sélectionnées
        query_params = {
            'selected_columns': selected_columns,
            'dataset': dataset_id,
            'model_name': model_name,
            'action': 'feature_selection'
        }
        return redirect(reverse('create_model_algo') + '?' + urlencode(query_params))

    dataset_id = request.GET.get('dataset')
    algorithm = request.GET.get('algorithm')
    k = int(request.GET.get('k', 10))
    num_features = int(request.GET.get('num_features', 10))

    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        return render(request, 'feature_selection_results.html', {'error_message': 'Dataset not found.'})

    # Récupérer les identifiants des colonnes sélectionnées au lieu des noms de colonnes
    selected_columns = request.GET.getlist('selected_columns')

    selected_features = run_feature_selection(dataset_id, algorithm, k, num_features)

    # Récupérer le nom du modèle précédemment choisi, si disponible
    model_name = request.session.get('model_name', 'Unknown Model')

    # Créer l'URL de retour avec les informations nécessaires
    return_url = reverse('create_model_algo') + f'?dataset={dataset_id}&model_name={model_name}'

    return render(request, 'feature_selection_results.html', {
        'dataset': dataset,
        'algorithm': algorithm,
        'selected_features': selected_features,
        'num_features': num_features,
        'return_url': return_url
    })



from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Dataset, Column, Model

"""
def create_model_algo(request):
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        dataset_id = request.POST.get('dataset')
        algorithm = request.POST.get('algorithm')
        selected_columns = request.POST.getlist('columns')

        # Enregistrer le modèle
        model = Model.objects.create(name=model_name, algorithm=algorithm, dataset_id=dataset_id)

        # Enregistrer les colonnes sélectionnées pour ce modèle avec le statut "input"
        for column_id in selected_columns:
            column = Column.objects.get(id=column_id)
            column.status = 'input'
            column.save()
            model.columns.add(column)

        messages.success(request, 'Le modèle a été enregistré avec succès.')
        return redirect('home')  # Rediriger vers la page d'accueil ou une autre vue après l'enregistrement du modèle

    # Pré-remplir les champs du formulaire avec les informations passées depuis la vue feature_selection_results
    dataset_id = request.GET.get('dataset')
    model_name = request.GET.get('model_name', 'Unknown Model')
    action = request.GET.get('action')
    selected_columns_id = request.GET.getlist('selected_columns')  # Récupérer les IDs des colonnes sélectionnées
    datasets = Dataset.objects.all()

    # Si l'action est feature_selection, récupérer les colonnes correspondantes
    if action == 'feature_selection':
        selected_columns = Column.objects.filter(id__in=selected_columns_id)
    else:
        selected_columns = []

    context = {
        'datasets': datasets,
        'model_name': model_name,
        'dataset_id': dataset_id,
        'selected_columns': selected_columns,
    }

    return render(request, 'create_model_algo.html', context)


def create_model_algo(request):
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        dataset_id = request.POST.get('dataset')
        algorithm = request.POST.get('algorithm')
        selected_columns = request.POST.getlist('columns')

        # Enregistrer le modèle
        model = Model.objects.create(name=model_name, algorithm=algorithm, dataset_id=dataset_id)

        # Enregistrer les colonnes sélectionnées pour ce modèle avec le statut "input"
        for column_id in selected_columns:
            column = Column.objects.get(id=column_id)
            column.status = 'input'
            column.save()
            model.columns.add(column)

        messages.success(request, 'Le modèle a été enregistré avec succès.')
        return redirect('home')  # Rediriger vers la page d'accueil ou une autre vue après l'enregistrement du modèle

    # Pré-remplir les champs du formulaire avec les informations passées depuis la vue feature_selection_results
    dataset_id = request.GET.get('dataset')
    model_name = request.GET.get('model_name', 'Unknown Model')
    action = request.GET.get('action')
    selected_columns_id = request.GET.getlist('selected_columns')  # Récupérer les IDs des colonnes sélectionnées
    datasets = Dataset.objects.all()

    # Si l'action est feature_selection, récupérer les colonnes correspondantes
    if action == 'feature_selection':
        # Récupérer les noms des colonnes à partir des identifiants
        selected_columns = Column.objects.filter(id__in=selected_columns_id).values_list('name', flat=True)
    else:
        selected_columns = []

    context = {
        'datasets': datasets,
        'model_name': model_name,
        'dataset_id': dataset_id,
        'selected_columns': selected_columns,
    }

    return render(request, 'create_model_algo.html', context)
"""

def create_model_algo(request):
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        dataset_id = request.POST.get('dataset')
        algorithm = request.POST.get('algorithm')
        selected_columns = request.POST.getlist('selected_columns')  # Obtenez les noms des colonnes, pas leurs ID

        # Enregistrer le modèle
        model = Model.objects.create(name=model_name, algorithm=algorithm, dataset_id=dataset_id)

        # Enregistrer les colonnes sélectionnées pour ce modèle avec le statut "input"
        for column_name in selected_columns:  # Utilisez column_name au lieu de column_id
            column = Column.objects.get(name=column_name)  # Recherchez la colonne par son nom
            column.status = 'input'
            column.save()
            model.columns.add(column)

        messages.success(request, 'Le modèle a été enregistré avec succès.')
        return redirect('home')  # Rediriger vers la page d'accueil ou une autre vue après l'enregistrement du modèle

    # Pré-remplir les champs du formulaire avec les informations passées depuis la vue feature_selection_results
    dataset_id = request.GET.get('dataset')
    model_name = request.GET.get('model_name', 'Unknown Model')
    action = request.GET.get('action')
    selected_columns = request.GET.getlist('selected_columns')  # Récupérer les noms des colonnes sélectionnées, pas leurs ID
    datasets = Dataset.objects.all()

    context = {
        'datasets': datasets,
        'model_name': model_name,
        'dataset_id': dataset_id,
        'selected_columns': selected_columns,
    }

    return render(request, 'create_model_algo.html', context)


@csrf_exempt
def get_columns(request, dataset_id):
    columns = Column.objects.filter(dataset_id=dataset_id, status='input').values('id', 'name')
    return JsonResponse({'columns': list(columns)})

