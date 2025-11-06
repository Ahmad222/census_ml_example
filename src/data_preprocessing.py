"""
Data preprocessing utilities for Census Income dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mode",
    missing_indicators: List[str] = ["?", " ?", "? ", " ? "]
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy to handle missing values: "mode", "median", "drop", "impute"
    missing_indicators : List[str]
        List of strings that indicate missing values
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with missing values handled
    """
    df = df.copy()
    
    # Replace missing indicators with NaN
    for indicator in missing_indicators:
        df = df.replace(indicator, np.nan)
    
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "mode":
        # Fill categorical with mode, numerical with median
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    elif strategy == "median":
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
    
    return df


def encode_categorical(
    df: pd.DataFrame,
    method: str = "onehot",
    columns: Optional[List[str]] = None,
    label_encoders: Optional[dict] = None
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Encode categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Encoding method: "onehot", "label", "target"
    columns : List[str], optional
        List of columns to encode. If None, encodes all object columns
    label_encoders : dict, optional
        Dictionary of label encoders for consistent encoding
        
    Returns:
    --------
    df : pd.DataFrame
        Encoded dataframe
    label_encoders : dict, optional
        Dictionary of label encoders used
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if method == "onehot":
        df = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=True)
        return df, None
    
    elif method == "label":
        if label_encoders is None:
            label_encoders = {}
        
        for col in columns:
            if col in df.columns:
                if col not in label_encoders:
                    label_encoders[col] = LabelEncoder()
                    df[col] = label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    known_classes = set(label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_classes else label_encoders[col].classes_[0]
                    )
                    df[col] = label_encoders[col].transform(df[col])
        
        return df, label_encoders
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def scale_numerical(
    df: pd.DataFrame,
    scaler_type: str = "standard",
    columns: Optional[List[str]] = None,
    scaler: Optional[object] = None
) -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    scaler_type : str
        Type of scaler: "standard", "minmax", "robust"
    columns : List[str], optional
        List of columns to scale. If None, scales all numerical columns
    scaler : object, optional
        Pre-fitted scaler for transform only
        
    Returns:
    --------
    df : pd.DataFrame
        Scaled dataframe
    scaler : object
        Fitted scaler
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if scaler is None:
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    
    return df, scaler


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    handle_missing_strategy: str = "mode",
    encode_method: str = "onehot",
    scale_numerical_features: bool = True,
    scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Complete data preparation pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target column
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
    handle_missing_strategy : str
        Strategy for handling missing values
    encode_method : str
        Method for encoding categorical variables
    scale_numerical_features : bool
        Whether to scale numerical features
    scaler_type : str
        Type of scaler to use
        
    Returns:
    --------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Test target
    preprocessors : dict
        Dictionary of preprocessors (encoders, scalers)
    """
    df = df.copy()
    
    # Separate target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Handle missing values
    X = handle_missing_values(X, strategy=handle_missing_strategy)
    
    # Encode categorical variables
    X, label_encoders = encode_categorical(X, method=encode_method)
    
    # Scale numerical features
    preprocessors = {"label_encoders": label_encoders}
    if scale_numerical_features:
        X, scaler = scale_numerical(X, scaler_type=scaler_type)
        preprocessors["scaler"] = scaler
    else:
        preprocessors["scaler"] = None
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessors

