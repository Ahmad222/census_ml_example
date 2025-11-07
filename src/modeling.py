"""
Modeling utilities for Census Income prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import lightgbm as lgb
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
from typing import Dict, Tuple, Optional, Any, List
import joblib
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    config_path: Optional[str] = None
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
        Hyperparameters for grid search. If None, will load from config file.
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    config_path : str, optional
        Path to model config YAML file. If provided, will load params from config.
        
    Returns:
    --------
    model : object
        Trained model
    results : dict
        Training results
    """
    if params is None:
        if config_path is not None:
            # Load from config file
            config = load_hyperparameter_config(config_path)
            if 'random_forest' in config.get('models', {}) and config['models']['random_forest'].get('enabled', True):
                params = config['models']['random_forest']['params'].copy()
            else:
                # Fallback to defaults
                params = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
        else:
            # Default params
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


def prepare_data_for_lightgbm(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    category_mappings: Optional[Dict[str, List]] = None
) -> Tuple[pd.DataFrame, List[int], Optional[Dict[str, List]]]:
    """
    Prepare data for LightGBM by identifying categorical and numerical features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    categorical_features : list, optional
        List of categorical feature names. If None, will auto-detect from dtypes.
    category_mappings : dict, optional
        Dictionary mapping column names to list of categories for consistent encoding.
        If provided, will use these mappings instead of creating new ones.
        
    Returns:
    --------
    X_processed : pd.DataFrame
        Features with categorical columns converted to integer codes
    categorical_indices : list
        List of column indices that are categorical
    category_mappings : dict
        Dictionary of category mappings (for use with validation/test sets)
    """
    X_processed = X.copy()
    
    # Identify categorical features
    if categorical_features is None:
        # Auto-detect: object dtype or boolean are categorical
        categorical_cols = []
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object' or X_processed[col].dtype == 'bool':
                categorical_cols.append(col)
    else:
        categorical_cols = [col for col in categorical_features if col in X_processed.columns]
    
    # Store category mappings if not provided (for consistent encoding across datasets)
    if category_mappings is None:
        category_mappings = {}
    
    # Convert categorical columns to integer codes (LightGBM requires integers for categorical)
    for col in categorical_cols:
        if X_processed[col].dtype == 'object' or str(X_processed[col].dtype) == 'object':
            try:
                if col in category_mappings:
                    # Use existing category mapping (for validation/test sets)
                    # First convert to categorical using the mapping
                    X_processed[col] = pd.Categorical(X_processed[col], categories=category_mappings[col], ordered=False)
                else:
                    # Create new category mapping (for training set)
                    unique_vals = X_processed[col].dropna().unique().tolist()
                    if len(unique_vals) == 0:
                        # If no unique values, create a dummy category
                        unique_vals = ['dummy']
                    category_mappings[col] = unique_vals
                    X_processed[col] = pd.Categorical(X_processed[col], categories=unique_vals, ordered=False)
                
                # Convert to codes and ensure int dtype
                codes = X_processed[col].cat.codes
                X_processed[col] = codes.astype(int)
                
                # Handle -1 (missing/unknown category) by setting to max+1
                if (X_processed[col] == -1).any():
                    max_code = X_processed[col].max()
                    X_processed[col] = X_processed[col].replace(-1, max_code + 1 if max_code >= 0 else 0).astype(int)
            except Exception as e:
                # Fallback: try direct numeric conversion
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0).astype(int)
        elif X_processed[col].dtype == 'bool' or str(X_processed[col].dtype) == 'bool':
            # Convert boolean to int
            X_processed[col] = X_processed[col].astype(int)
    
    # Ensure numerical columns are numeric
    numerical_cols = [col for col in X_processed.columns if col not in categorical_cols]
    for col in numerical_cols:
        if not pd.api.types.is_numeric_dtype(X_processed[col]):
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)
    
    # Final check: ensure all columns are numeric (int, float, or bool)
    # Convert any remaining non-numeric columns
    for col in X_processed.columns:
        dtype_name = str(X_processed[col].dtype)
        if X_processed[col].dtype == 'object' or dtype_name == 'object':
            # If still object, try to convert to numeric
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0).astype(float)
        elif 'category' in dtype_name.lower():
            # If still category dtype, convert to int
            if hasattr(X_processed[col], 'cat'):
                X_processed[col] = X_processed[col].cat.codes.astype(int)
            else:
                X_processed[col] = X_processed[col].astype(int)
    
    # Verify all columns are now numeric
    for col in X_processed.columns:
        if not pd.api.types.is_numeric_dtype(X_processed[col]):
            # Force conversion to float as last resort
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0).astype(float)
    
    # Get categorical feature indices (column positions)
    categorical_indices = [i for i, col in enumerate(X_processed.columns) if col in categorical_cols]
    
    return X_processed, categorical_indices, category_mappings


def load_hyperparameter_config(config_path: str = "config/model_config.yaml") -> Dict:
    """
    Load hyperparameter configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to model config YAML file
        
    Returns:
    --------
    config : dict
        Dictionary containing hyperparameter configurations
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optuna_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_config: Dict,
    class_weight: Optional[Dict] = None
):
    """
    Create Optuna objective function for LightGBM hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    param_config : dict
        Hyperparameter configuration from config file
    class_weight : dict, optional
        Class weights for handling imbalance
        
    Returns:
    --------
    objective : callable
        Optuna objective function
    """
    def objective(trial):
        # Suggest hyperparameters based on config
        params = {}
        for param_name, param_spec in param_config.items():
            if param_spec['type'] == 'int':
                if param_spec.get('log', False):
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_spec['low'],
                        param_spec['high']
                    )
            elif param_spec['type'] == 'float':
                if param_spec.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_spec['low'],
                        param_spec['high']
                    )
        
        # Add fixed parameters
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['boosting_type'] = 'gbdt'
        params['verbose'] = -1
        params['random_state'] = 42
        
        # Calculate scale_pos_weight for class imbalance (LightGBM uses this instead of class_weight)
        if class_weight is not None:
            # Calculate ratio: negative_class_weight / positive_class_weight
            # For binary classification, this is typically: n_positive / n_negative
            n_positive = (y_train == 1).sum()
            n_negative = (y_train == 0).sum()
            if n_positive > 0 and n_negative > 0:
                params['scale_pos_weight'] = n_negative / n_positive
        
        # Prepare data for LightGBM (identify categorical and numerical features)
        # First, identify categorical columns from training data
        X_train_processed, categorical_indices, category_mappings = prepare_data_for_lightgbm(X_train)
        
        # Get categorical column names from training data
        categorical_cols = [X_train.columns[i] for i in categorical_indices]
        
        # Process validation set using same categorical columns and mappings
        X_val_processed, _, _ = prepare_data_for_lightgbm(
            X_val, 
            categorical_features=categorical_cols,
            category_mappings=category_mappings
        )
        
        # Final verification: ensure all columns are numeric before creating Dataset
        for col in X_train_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_train_processed[col]):
                X_train_processed[col] = pd.to_numeric(X_train_processed[col], errors='coerce').fillna(0).astype(float)
        for col in X_val_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_val_processed[col]):
                X_val_processed[col] = pd.to_numeric(X_val_processed[col], errors='coerce').fillna(0).astype(float)
        
        # Create LightGBM datasets with categorical feature specification
        train_data = lgb.Dataset(
            X_train_processed,
            label=y_train,
            categorical_feature=categorical_indices if categorical_indices else None,
            free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val_processed,
            label=y_val,
            reference=train_data,
            categorical_feature=categorical_indices if categorical_indices else None,
            free_raw_data=False
        )
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['val'],
            num_boost_round=params.get('n_estimators', 100),
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Predict on validation set (use processed version)
        y_pred_proba = model.predict(X_val_processed, num_iteration=model.best_iteration)
        
        # Calculate AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        
        return auc
    
    return objective


def optimize_lightgbm_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config_path: str = "config/model_config.yaml",
    n_trials_random: int = 20,
    n_trials_tpe: int = 50,
    random_state: int = 42
) -> Tuple[optuna.Study, Dict]:
    """
    Optimize LightGBM hyperparameters using Optuna.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    config_path : str
        Path to model config YAML file
    n_trials_random : int
        Number of random search trials
    n_trials_tpe : int
        Number of TPE optimization trials
    random_state : int
        Random seed
        
    Returns:
    --------
    study : optuna.Study
        Optuna study object
    best_params : dict
        Best hyperparameters found
    """
    # Load hyperparameter configuration
    config = load_hyperparameter_config(config_path)
    param_config = config['models']['lightgbm']['params']
    
    # Calculate class weights for imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Create objective function
    objective = create_optuna_objective(
        X_train, y_train, X_val, y_val, param_config, class_weight_dict
    )
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='lightgbm_optimization',
        sampler=optuna.samplers.RandomSampler(seed=random_state)
    )
    
    # Run random search trials
    print(f"Running {n_trials_random} random search trials...")
    study.optimize(objective, n_trials=n_trials_random, show_progress_bar=True)
    
    # Switch to TPE sampler for remaining trials
    study.sampler = optuna.samplers.TPESampler(seed=random_state)
    print(f"Running {n_trials_tpe} TPE optimization trials...")
    study.optimize(objective, n_trials=n_trials_tpe, show_progress_bar=True)
    
    best_params = study.best_params.copy()
    
    return study, best_params


def train_lightgbm_with_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict,
    class_weight: Optional[Dict] = None
) -> lgb.Booster:
    """
    Train LightGBM model with given hyperparameters.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    params : dict
        Hyperparameters for LightGBM
    class_weight : dict, optional
        Class weights for handling imbalance
        
    Returns:
    --------
    model : lgb.Booster
        Trained LightGBM model
    """
    # Prepare parameters
    model_params = params.copy()
    model_params['objective'] = 'binary'
    model_params['metric'] = 'auc'
    model_params['boosting_type'] = 'gbdt'
    model_params['verbose'] = -1
    model_params['random_state'] = 42
    
    # Calculate scale_pos_weight for class imbalance (LightGBM uses this)
    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()
    if n_positive > 0 and n_negative > 0:
        model_params['scale_pos_weight'] = n_negative / n_positive
    
    # Prepare data for LightGBM (identify categorical and numerical features)
    # First, identify categorical columns from training data
    X_train_processed, categorical_indices, category_mappings = prepare_data_for_lightgbm(X_train)
    
    # Get categorical column names from training data
    categorical_cols = [X_train.columns[i] for i in categorical_indices]
    
    # Process validation set using same categorical columns and mappings
    X_val_processed, _, _ = prepare_data_for_lightgbm(
        X_val, 
        categorical_features=categorical_cols,
        category_mappings=category_mappings
    )
    
    # Create datasets with categorical feature specification
    train_data = lgb.Dataset(
        X_train_processed,
        label=y_train,
        categorical_feature=categorical_indices if categorical_indices else None,
        free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val_processed,
        label=y_val,
        reference=train_data,
        categorical_feature=categorical_indices if categorical_indices else None,
        free_raw_data=False
    )
    
    # Train model
    model = lgb.train(
        model_params,
        train_data,
        valid_sets=[val_data],
        valid_names=['val'],
        num_boost_round=model_params.get('n_estimators', 100),
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    return model


def evaluate_model_performance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Dataset"
) -> Dict:
    """
    Evaluate model performance and return metrics.
    
    Parameters:
    -----------
    model : object
        Trained model (LightGBM Booster or sklearn-compatible)
    X : pd.DataFrame
        Features
    y : pd.Series
        True labels
    dataset_name : str
        Name of the dataset (for display)
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Predict
    if isinstance(model, lgb.Booster):
        # Prepare data for LightGBM (identify categorical and numerical features)
        X_processed, _, _ = prepare_data_for_lightgbm(X)
        y_pred_proba = model.predict(X_processed, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'dataset': dataset_name,
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y, y_pred_proba)
    metrics['precision_recall_curve'] = (precision, recall, pr_thresholds)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y, y_pred_proba)
    metrics['roc_curve'] = (fpr, tpr, roc_thresholds)
    
    return metrics


def plot_model_performance(
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot model performance across train, validation, and test sets.
    
    Parameters:
    -----------
    train_metrics : dict
        Training set metrics
    val_metrics : dict
        Validation set metrics
    test_metrics : dict
        Test set metrics
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Metrics comparison bar plot
    metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    metrics_to_plot = ['precision', 'recall', 'f1', 'roc_auc']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, (name, metrics) in enumerate([('Train', train_metrics), 
                                          ('Val', val_metrics), 
                                          ('Test', test_metrics)]):
        values = [metrics[m] for m in metrics_to_plot]
        axes[0, 0].bar(x + i*width, values, width, label=name, alpha=0.8)
    
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Metrics')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # 2. ROC curves
    for name, metrics in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
        fpr, tpr, _ = metrics['roc_curve']
        axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC = {metrics['roc_auc']:.3f})", linewidth=2)
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Precision-Recall curves
    for name, metrics in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
        precision, recall, _ = metrics['precision_recall_curve']
        axes[1, 0].plot(recall, precision, label=f"{name}", linewidth=2)
    
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Confusion matrix (Test set)
    cm = test_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=True)
    axes[1, 1].set_title('Confusion Matrix - Test Set')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

