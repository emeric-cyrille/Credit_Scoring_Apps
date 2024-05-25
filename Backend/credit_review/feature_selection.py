
import os
import pandas as pd
from .models import Dataset, Column

def run_feature_selection(dataset_id, algorithm, k=None, num_features=10):
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        print(f"Dataset with id {dataset_id} does not exist.")
        return []

    try:
        # Récupérer toutes les colonnes associées à ce dataset
        columns = Column.objects.filter(dataset=dataset)

        # Filtrer la colonne avec le statut 'target'
        target_column_obj = columns.filter(status='target').first()
        if not target_column_obj:
            print(f"Target column not found in dataset.")
            return []

        target_column = target_column_obj.name

        # Construire le chemin complet du fichier
        file_path = dataset.data.path

        data = pd.read_csv(file_path)

        # Sélectionner les colonnes qui ne sont pas de type objet
        numeric_columns = data.select_dtypes(exclude=['object']).columns

        # Supprimer les colonnes non numériques de X
        X = data[numeric_columns]

        # Convertir les colonnes catégorielles en variables binaires avec get_dummies
        categorical_columns = data.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_columns)

        # Séparer la variable cible y
        y = data[target_column]

        # Exclure la colonne cible des colonnes sélectionnées
        if target_column in X.columns:
            X.drop(columns=[target_column], inplace=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    try:
        if algorithm == 'xgboost':
            significant_columns = run_xgboost_selection(X, y)
        elif algorithm == 'sfs':
            significant_columns = run_sfs_selection(X, y, k)
        elif algorithm == 'lasso':
            significant_columns = run_lasso_selection(X, y, k)
        elif algorithm == 'relief':
            significant_columns = run_relief_selection(X, y)
        else:
            print(f"Unknown algorithm '{algorithm}'.")
            significant_columns = []
    except Exception as e:
        print(f"Error running feature selection with algorithm '{algorithm}': {e}")
        return []

    if not significant_columns:
        print(f"No significant columns found using algorithm '{algorithm}' on dataset '{dataset_id}'.")
    else:
        print(f"Significant columns found: {significant_columns}")

    # Limiter le nombre d'attributs retournés à num_features
    return significant_columns[:num_features]

def run_xgboost_selection(X, y):

    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entraînement du modèle XGBoost
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Récupération de l'importance des attributs
    feature_importance = model.feature_importances_

    # Création d'une série avec l'importance des attributs
    importance_series = pd.Series(feature_importance, index=X_train.columns)

    # Trier les attributs par ordre décroissant de leur importance
    sorted_features = importance_series.sort_values(ascending=False)

    # Calculer le pourcentage de participation de chaque attribut
    total_importance = sorted_features.sum()
    percentages = (sorted_features / total_importance) * 100

    # Création d'une liste de tuples (nom de la colonne, pourcentage de participation)
    selected_features = [(feature, percentages[feature]) for feature in sorted_features.index]

    return selected_features


"""def run_sfs_selection(X, y, k):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fonction pour calculer les pourcentages de participation des attributs
    def calculate_percentage_importance(features, importances):
        total_importance = sum(importances)
        percentages = [(importance / total_importance) * 100 for importance in importances]
        return percentages

    # Algorithme Sequential Forward Selection (SFS)
    def SFS(X_train, y_train, n_features):
        selected_features = []
        selected_percentages = []
        while len(selected_features) < n_features:
            best_feature = None
            best_score = -1
            for feature in X_train.columns:
                if feature not in selected_features:
                    temp_features = selected_features + [feature]
                    X_temp = X_train[temp_features]
                    model = KNeighborsClassifier()
                    model.fit(X_temp, y_train)
                    y_pred = model.predict(X_temp)
                    score = accuracy_score(y_train, y_pred)
                    if score > best_score:
                        best_score = score
                        best_feature = feature
            selected_features.append(best_feature)
            selected_percentages.append(best_score)
        return selected_features, selected_percentages

    # Exécution de l'algorithme SFS
    selected_features_sfs, selected_percentages_sfs = SFS(X_train, y_train, k)

    # Création de la liste de tuples (nom de la colonne, pourcentage de participation)
    selected_features = [(feature, percentage) for feature, percentage in
                         zip(selected_features_sfs, selected_percentages_sfs)]

    return selected_features"""

def run_sfs_selection(X, y, k):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fonction pour calculer les pourcentages de participation des attributs
    def calculate_percentage_importance(features, importances):
        total_importance = sum(importances)
        percentages = [(importance / total_importance) * 100 for importance in importances]
        return percentages

    # Algorithme Sequential Forward Selection (SFS)
    def SFS(X_train, y_train, n_features):
        selected_features = []
        selected_percentages = []
        while len(selected_features) < n_features:
            best_feature = None
            best_score = -1
            for feature in X_train.columns:
                if feature not in selected_features:
                    temp_features = selected_features + [feature]
                    X_temp = X_train[temp_features]
                    model = KNeighborsClassifier()
                    model.fit(X_temp, y_train)
                    y_pred = model.predict(X_temp)
                    score = accuracy_score(y_train, y_pred)
                    if score > best_score:
                        best_score = score
                        best_feature = feature
            selected_features.append(best_feature)
            selected_percentages.append(best_score)

        # Création de la liste de tuples (nom de la colonne, pourcentage de participation)
        selected_features_with_percentages = [(feature, percentage) for feature, percentage in
                                              zip(selected_features, selected_percentages)]

        # Trier les caractéristiques par ordre décroissant de leur pourcentage de participation
        sorted_features = sorted(selected_features_with_percentages, key=lambda x: x[1], reverse=True)

        return sorted_features

    # Exécution de l'algorithme SFS
    selected_features_sfs = SFS(X_train, y_train, k)

    return selected_features_sfs

def run_lasso_selection(X, y, k):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LassoCV

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entraînement du modèle LASSO
    lasso = LassoCV(cv=5)
    lasso.fit(X_train, y_train)

    # Récupération des coefficients de LASSO
    coef = pd.Series(lasso.coef_, index=X_train.columns)

    # Sélection des k caractéristiques les plus importantes (celles dont les coefficients sont non nuls)
    selected_features = coef[coef != 0].index.tolist()[:k]

    # Calcul du pourcentage de participation de chaque attribut
    total_importance = coef.abs().sum()
    percentages = (coef.abs() / total_importance) * 100
    selected_features_with_percentages = [(feature, percentages[feature]) for feature in selected_features]

    # Trier les caractéristiques par ordre décroissant de leur pourcentage de participation
    selected_features_sorted = sorted(selected_features_with_percentages, key=lambda x: x[1], reverse=True)

    return selected_features_sorted



"""
def run_lasso_selection(X, y, k):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LassoCV

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entraînement du modèle LASSO
    lasso = LassoCV(cv=5)
    lasso.fit(X_train, y_train)

    # Récupération des coefficients de LASSO
    coef = pd.Series(lasso.coef_, index=X_train.columns)

    # Sélection des k caractéristiques les plus importantes (celles dont les coefficients sont non nuls)
    selected_features = coef[coef != 0].index.tolist()[:k]

    # Calcul du pourcentage de participation de chaque attribut
    total_importance = coef.abs().sum()
    percentages = (coef.abs() / total_importance) * 100
    selected_features_with_percentages = [(feature, percentages[feature]) for feature in selected_features]

    return selected_features_with_percentages
"""
def run_relief_selection(X, y, sample_size=1000):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from skrebate import ReliefF

    # Sélectionner un échantillon aléatoire de données pour l'entraînement du modèle ReliefF
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size, random_state=42)

    # Entraînement de l'algorithme ReliefF
    relief = ReliefF()
    relief.fit(X_sample.values, y_sample.values)

    # Récupération de l'importance des caractéristiques
    feature_importance = relief.feature_importances_

    # Création d'une série avec l'importance des caractéristiques
    importance_series = pd.Series(feature_importance, index=X.columns)

    # Trier les caractéristiques par ordre décroissant de leur importance
    sorted_features = importance_series.sort_values(ascending=False)

    # Calculer le pourcentage de participation de chaque caractéristique
    total_importance = sorted_features.sum()
    percentages = (sorted_features / total_importance) * 100

    # Création d'une liste de tuples (nom de la caractéristique, pourcentage de participation)
    selected_features = [(feature, percentages[feature]) for feature in sorted_features.index]

    # Trier les caractéristiques par ordre décroissant de leur pourcentage de participation
    selected_features_sorted = sorted(selected_features, key=lambda x: x[1], reverse=True)

    return selected_features_sorted


"""
def run_relief_selection(X, y, sample_size=1000):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from skrebate import ReliefF

    # Sélectionner un échantillon aléatoire de données pour l'entraînement du modèle ReliefF
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size, random_state=42)

    # Entraînement de l'algorithme ReliefF
    relief = ReliefF()
    relief.fit(X_sample.values, y_sample.values)

    # Récupération de l'importance des caractéristiques
    feature_importance = relief.feature_importances_

    # Création d'une série avec l'importance des caractéristiques
    importance_series = pd.Series(feature_importance, index=X.columns)

    # Trier les caractéristiques par ordre décroissant de leur importance
    sorted_features = importance_series.sort_values(ascending=False)

    # Calculer le pourcentage de participation de chaque caractéristique
    total_importance = sorted_features.sum()
    percentages = (sorted_features / total_importance) * 100

    # Création d'une liste de tuples (nom de la caractéristique, pourcentage de participation)
    selected_features = [(feature, percentages[feature]) for feature in sorted_features.index]

    return selected_features
    
    """


