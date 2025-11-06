"""
Feature engineering utilities for Census Income dataset.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    chi2
)
from typing import Tuple, Optional, List


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Create interaction features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_pairs : List[Tuple[str, str]], optional
        List of feature pairs to create interactions for.
        If None, creates interactions for all numerical features.
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with interaction features
    """
    df = df.copy()
    
    if feature_pairs is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Create interactions for top features (limit to avoid explosion)
        if len(numerical_cols) > 10:
            numerical_cols = numerical_cols[:10]
        feature_pairs = [
            (numerical_cols[i], numerical_cols[j])
            for i in range(len(numerical_cols))
            for j in range(i + 1, len(numerical_cols))
        ]
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            if df[feat1].dtype in [np.number] and df[feat2].dtype in [np.number]:
                df[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]
    
    return df


def create_polynomial_features(
    df: pd.DataFrame,
    degree: int = 2,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create polynomial features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    degree : int
        Degree of polynomial features
    columns : List[str], optional
        List of columns to create polynomial features for.
        If None, uses all numerical columns.
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with polynomial features
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns:
            for d in range(2, degree + 1):
                df[f"{col}_pow_{d}"] = df[col] ** d
    
    return df


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "mutual_info",
    k: int = 20,
    score_func: Optional[object] = None
) -> Tuple[pd.DataFrame, object]:
    """
    Select top k features using statistical tests.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target series
    method : str
        Feature selection method: "mutual_info", "chi2", "f_classif"
    k : int
        Number of features to select
    score_func : object, optional
        Custom scoring function
        
    Returns:
    --------
    X_selected : pd.DataFrame
        Dataframe with selected features
    selector : object
        Fitted feature selector
    """
    if score_func is None:
        if method == "mutual_info":
            score_func = mutual_info_classif
        elif method == "chi2":
            score_func = chi2
        elif method == "f_classif":
            score_func = f_classif
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    # Limit k to number of features
    k = min(k, X.shape[1])
    
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    return X_selected, selector


def engineer_features(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    create_interactions: bool = False,
    create_polynomial: bool = False,
    polynomial_degree: int = 2,
    feature_selection: bool = False,
    selection_method: str = "mutual_info",
    k_features: int = 20
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, optional
        Name of target column (needed for feature selection)
    create_interactions : bool
        Whether to create interaction features
    create_polynomial : bool
        Whether to create polynomial features
    polynomial_degree : int
        Degree of polynomial features
    feature_selection : bool
        Whether to perform feature selection
    selection_method : str
        Feature selection method
    k_features : int
        Number of features to select
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with engineered features
    selector : object, optional
        Feature selector if feature selection was performed
    """
    df = df.copy()
    
    # Separate target if provided
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        X = df
    
    # Create interaction features
    if create_interactions:
        X = create_interaction_features(X)
    
    # Create polynomial features
    if create_polynomial:
        X = create_polynomial_features(X, degree=polynomial_degree)
    
    # Feature selection
    selector = None
    if feature_selection and y is not None:
        X, selector = select_features(X, y, method=selection_method, k=k_features)
    
    # Rejoin target if it was separated
    if y is not None:
        df = pd.concat([X, y], axis=1)
    else:
        df = X
    
    return df, selector

