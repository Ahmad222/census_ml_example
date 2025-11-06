"""
Modeling utilities for Census Income prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, Optional, Any
import joblib
from pathlib import Path


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "roc_auc"
) -> Tuple[Any, Dict]:
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    params : dict, optional
        Hyperparameters for grid search
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
        
    Returns:
    --------
    model : object
        Trained model
    results : dict
        Training results
    """
    if params is None:
        params = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
    
    base_model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        base_model, params, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return grid_search.best_estimator_, results


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "roc_auc"
) -> Tuple[Any, Dict]:
    """
    Train Random Forest model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    params : dict, optional
        Hyperparameters for grid search
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
        
    Returns:
    --------
    model : object
        Trained model
    results : dict
        Training results
    """
    if params is None:
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = RandomizedSearchCV(
        base_model, params, cv=cv, scoring=scoring, n_jobs=-1, verbose=1, n_iter=20
    )
    grid_search.fit(X_train, y_train)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return grid_search.best_estimator_, results


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "roc_auc"
) -> Tuple[Any, Dict]:
    """
    Train XGBoost model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    params : dict, optional
        Hyperparameters for grid search
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
        
    Returns:
    --------
    model : object
        Trained model
    results : dict
        Training results
    """
    if params is None:
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    
    base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    grid_search = RandomizedSearchCV(
        base_model, params, cv=cv, scoring=scoring, n_jobs=-1, verbose=1, n_iter=20
    )
    grid_search.fit(X_train, y_train)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return grid_search.best_estimator_, results


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "roc_auc"
) -> Tuple[Any, Dict]:
    """
    Train LightGBM model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    params : dict, optional
        Hyperparameters for grid search
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
        
    Returns:
    --------
    model : object
        Trained model
    results : dict
        Training results
    """
    if params is None:
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'feature_fraction': [0.8, 0.9, 1.0]
        }
    
    base_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    grid_search = RandomizedSearchCV(
        base_model, params, cv=cv, scoring=scoring, n_jobs=-1, verbose=1, n_iter=20
    )
    grid_search.fit(X_train, y_train)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return grid_search.best_estimator_, results


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            metrics['roc_auc'] = None
    
    return metrics


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model names and trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        DataFrame with model comparison metrics
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    return comparison_df

