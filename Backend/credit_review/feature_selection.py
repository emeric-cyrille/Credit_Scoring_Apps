import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier
from django.shortcuts import get_object_or_404
from skrebate import ReliefF  # Utilisation de skrebate pour ReliefF
from .models import Dataset

def run_feature_selection(dataset_id, algorithm, k=None):
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        return []

    try:
        data = pd.read_csv(dataset.file_path)
        X = data.drop(columns=[dataset.target_column])
        y = data[dataset.target_column]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    if dataset.target_column not in data.columns:
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
            significant_columns = []
    except Exception as e:
        print(f"Error running feature selection: {e}")
        return []

    return significant_columns

def run_xgboost_selection(X, y):
    model = XGBClassifier()
    model.fit(X, y)
    importance = model.feature_importances_
    threshold = np.mean(importance)
    selection = SelectFromModel(model, threshold=threshold, prefit=True)
    selected_features = X.columns[selection.get_support(indices=True)].tolist()
    return selected_features

def run_sfs_selection(X, y, k):
    model = XGBClassifier()
    sfs = SequentialFeatureSelector(model, n_features_to_select=int(k), direction='forward')
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support(indices=True)].tolist()
    return selected_features

def run_lasso_selection(X, y, k):
    lasso = LassoCV(cv=5)
    lasso.fit(X, y)
    model = SelectFromModel(lasso, max_features=int(k))
    model.fit(X, y)
    selected_features = X.columns[model.get_support(indices=True)].tolist()
    return selected_features

def run_relief_selection(X, y):
    relief = ReliefF(n_neighbors=100)
    relief.fit(X.to_numpy(), y.to_numpy())
    importance = relief.feature_importances_
    threshold = np.mean(importance)
    selected_features = X.columns[importance > threshold].tolist()
    return selected_features
