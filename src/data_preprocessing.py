"""
Data preprocessing utilities for Census Income dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mode",
    missing_indicators: List[str] = ["?", " ?", "? ", " ? "],
    categorical_fill_value: str = "not identified",
    numerical_fill_value: Optional[float] = None
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
    categorical_fill_value : str
        Value to use for filling missing categorical values (default: "not identified")
    numerical_fill_value : float, optional
        Value to use for filling missing numerical values. If None, uses median (default: None)
        
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
        # Fill categorical with specified value, numerical with specified value or median
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(categorical_fill_value, inplace=True)
            else:
                if numerical_fill_value is not None:
                    df[col].fillna(numerical_fill_value, inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
    elif strategy == "median":
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if numerical_fill_value is not None:
                    df[col].fillna(numerical_fill_value, inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(categorical_fill_value, inplace=True)
    
    return df


def encode_categorical(
    df: pd.DataFrame,
    method: str = "onehot",
    columns: Optional[List[str]] = None,
    label_encoders: Optional[dict] = None,
    frequency_encoders: Optional[dict] = None,
    max_categories_for_onehot: int = 5
) -> Tuple[pd.DataFrame, Optional[dict], Optional[dict]]:
    """
    Encode categorical variables.
    
    Uses one-hot encoding for categories with <= max_categories_for_onehot unique values,
    and frequency encoding for categories with more unique values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Encoding method: "onehot", "label", "target", "hybrid" (default: hybrid for method="onehot")
        - "hybrid": one-hot for <= max_categories_for_onehot, frequency for others
        - "onehot": one-hot encoding for all (legacy behavior)
        - "label": label encoding for all
    columns : List[str], optional
        List of columns to encode. If None, encodes all object columns
    label_encoders : dict, optional
        Dictionary of label encoders for consistent encoding (used with method="label")
    frequency_encoders : dict, optional
        Dictionary of frequency mappings for consistent encoding (used with method="hybrid")
    max_categories_for_onehot : int
        Maximum number of unique categories to use one-hot encoding (default: 5)
        
    Returns:
    --------
    df : pd.DataFrame
        Encoded dataframe
    label_encoders : dict, optional
        Dictionary of label encoders used (if method="label")
    frequency_encoders : dict, optional
        Dictionary of frequency mappings used (if method="hybrid" or "onehot")
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if method == "onehot" or method == "hybrid":
        # Use hybrid approach: one-hot for low cardinality, frequency for high cardinality
        if frequency_encoders is None:
            frequency_encoders = {}
        
        # Identify columns for one-hot vs frequency encoding
        onehot_columns = []
        frequency_columns = []
        
        for col in columns:
            if col in df.columns:
                n_unique = df[col].nunique()
                if n_unique <= max_categories_for_onehot:
                    onehot_columns.append(col)
                else:
                    frequency_columns.append(col)
        
        # Apply one-hot encoding to low cardinality columns
        if onehot_columns:
            df = pd.get_dummies(df, columns=onehot_columns, drop_first=False)
        
        # Apply frequency encoding to high cardinality columns
        for col in frequency_columns:
            if col in df.columns:
                if col not in frequency_encoders:
                    # Calculate frequency mapping from training data
                    freq_map = df[col].value_counts().to_dict()
                    frequency_encoders[col] = freq_map
                else:
                    # Use existing frequency mapping (for test/validation sets)
                    freq_map = frequency_encoders[col]
                
                # Replace categories with their frequencies
                # For unseen categories, use 0 or the minimum frequency
                df[col] = df[col].map(freq_map)
                if df[col].isna().any():
                    # Handle unseen categories: use minimum frequency or 0
                    min_freq = min(freq_map.values()) if freq_map else 0
                    df[col] = df[col].fillna(min_freq)
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df, None, frequency_encoders
    
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
        
        return df, label_encoders, None
    
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
    X, label_encoders, frequency_encoders = encode_categorical(X, method=encode_method)
    
    # Scale numerical features
    preprocessors = {"label_encoders": label_encoders, "frequency_encoders": frequency_encoders}
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


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    subset : List[str], optional
        List of column names to consider for duplicate detection.
        If None, considers all columns.
    keep : str
        Which duplicates to keep: 'first', 'last', or False (drop all)
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with duplicates removed
    """
    df = df.copy()
    
    if subset is None:
        # Remove duplicates across all columns
        df = df.drop_duplicates(keep=keep)
    else:
        # Remove duplicates based on subset of columns
        df = df.drop_duplicates(subset=subset, keep=keep)
    
    return df


def treat_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'winsorize',
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[pd.DataFrame, Optional[Dict[str, Tuple[float, float]]]]:
    """
    Treat outliers in numerical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        List of numerical columns to treat. If None, treats all numerical columns.
    method : str
        Method to treat outliers: 'winsorize', 'clip', 'remove'
    lower_percentile : float
        Lower percentile for winsorize/clip (default: 0.01)
    upper_percentile : float
        Upper percentile for winsorize/clip (default: 0.99)
    bounds : Dict[str, Tuple[float, float]], optional
        Pre-computed bounds dictionary {column_name: (lower_bound, upper_bound)}.
        If provided, uses these bounds instead of calculating from data.
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with outliers treated
    bounds : Dict[str, Tuple[float, float]]
        Dictionary of bounds used for each column (for reuse on test data)
    """
    df = df.copy()
    
    if columns is None:
        # Get all numerical columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    computed_bounds = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'winsorize' or method == 'clip':
            # Use pre-computed bounds if provided, otherwise calculate from data
            if bounds and col in bounds:
                lower_bound, upper_bound = bounds[col]
            else:
                # Calculate percentiles from current data
                lower_bound = df[col].quantile(lower_percentile)
                upper_bound = df[col].quantile(upper_percentile)
                computed_bounds[col] = (lower_bound, upper_bound)
            
            # Clip values to bounds
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'remove':
            # Remove rows with outliers (using IQR method)
            if bounds and col in bounds:
                lower_bound, upper_bound = bounds[col]
            else:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                computed_bounds[col] = (lower_bound, upper_bound)
            
            # Remove outliers
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Return bounds if they were computed (for reuse on test data)
    return_bounds = bounds if bounds else computed_bounds if computed_bounds else None
    
    return df, return_bounds


def make_label_binary(
    df: pd.DataFrame,
    target_column: str,
    positive_class: str = '50000+.'
) -> pd.DataFrame:
    """
    Convert target label to binary (0, 1).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target column
    positive_class : str
        Value that should be mapped to 1
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with binary target
    """
    df = df.copy()
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Convert to string and strip whitespace for robust comparison
    df[target_column] = df[target_column].astype(str).str.strip()
    positive_class = str(positive_class).strip()
    
    # Check what unique values exist in the target column
    unique_values = df[target_column].unique()
    
    # Verify that positive_class exists in the data
    if positive_class not in unique_values:
        raise ValueError(
            f"Positive class '{positive_class}' not found in target column. "
            f"Available values: {unique_values.tolist()}"
        )
    
    # Map to binary: positive_class -> 1, others -> 0
    df[target_column] = (df[target_column] == positive_class).astype(int)
    
    return df


def feature_engineering(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, optional
        Name of target column (to exclude from feature engineering)
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    # Exclude target from feature engineering
    feature_cols = [col for col in df.columns if col != target_column]
    
    # 1. Create binary flags for financial features
    if 'capital_gains' in df.columns:
        df['has_capital_gains'] = (df['capital_gains'] > 0).astype(int)
    
    if 'capital_losses' in df.columns:
        df['has_capital_losses'] = (df['capital_losses'] > 0).astype(int)
    
    if 'dividends_from_stocks' in df.columns:
        df['has_dividends'] = (df['dividends_from_stocks'] > 0).astype(int)
    
    if 'wage_per_hour' in df.columns:
        df['has_wage'] = (df['wage_per_hour'] > 0).astype(int)
        # Handle special value 9999 (likely missing/unknown)
        df['wage_per_hour'] = df['wage_per_hour'].replace(9999, np.nan)
    
    # 2. Create total financial assets (if applicable)
    financial_cols = ['capital_gains', 'capital_losses', 'dividends_from_stocks']
    if all(col in df.columns for col in financial_cols):
        df['total_financial_assets'] = (
            df['capital_gains'] - df['capital_losses'] + df['dividends_from_stocks']
        )
    
    # 3. Create work intensity feature
    if 'weeks_worked_in_year' in df.columns and 'num_persons_worked_for_employer' in df.columns:
        df['work_intensity'] = (
            df['weeks_worked_in_year'] * df['num_persons_worked_for_employer']
        )
    
    return df


def split_train_val_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe (from learn CSV)
    test_df : pd.DataFrame
        Test dataframe (from test CSV)
    target_column : str
        Name of target column
    val_size : float
        Proportion of validation set from training data
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training target
    y_val : pd.Series
        Validation target
    y_test : pd.Series
        Test target
    """
    # Separate features and target
    X_train_full = train_df.drop(columns=[target_column])
    y_train_full = train_df[target_column]
    
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def select_features_by_importance(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    importance_path: str,
    n_features: int = 30,
    target_column: Optional[str] = None,
    engineered_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top features based on feature importance from EDA, plus all engineered features.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    importance_path : str
        Path to CSV file with feature importance (from EDA)
    n_features : int
        Number of top features to select from importance file
    target_column : str, optional
        Name of target column (will always be included)
    engineered_features : List[str], optional
        List of engineered feature names to always include. If None, uses default list:
        ['has_capital_gains', 'has_capital_losses', 'has_dividends', 'has_wage',
         'total_financial_assets', 'work_intensity']
        
    Returns:
    --------
    train_df_selected : pd.DataFrame
        Training dataframe with selected features
    test_df_selected : pd.DataFrame
        Test dataframe with selected features
    """
    import pandas as pd
    from pathlib import Path
    
    # Default engineered features (created during feature engineering step)
    if engineered_features is None:
        engineered_features = [
            'has_capital_gains',
            'has_capital_losses',
            'has_dividends',
            'has_wage',
            'total_financial_assets',
            'work_intensity'
        ]
    
    # Load feature importance
    importance_df = pd.read_csv(importance_path)
    
    # Get top N features from importance file
    top_features = importance_df.head(n_features)['feature'].tolist()
    
    # Always include engineered features (if they exist in the dataframe)
    available_engineered = [f for f in engineered_features if f in train_df.columns]
    
    # Always include target column if specified
    if target_column and target_column in train_df.columns:
        if target_column not in top_features:
            top_features.append(target_column)
    
    # Combine: top features from importance + engineered features
    selected_features = list(set(top_features + available_engineered))
    
    # Select features (only those that exist in the dataframe)
    available_features = [f for f in selected_features if f in train_df.columns]
    
    print(f"\nFeature Selection based on EDA importance:")
    print(f"  Total features in importance file: {len(importance_df)}")
    print(f"  Requested top features from importance: {n_features}")
    print(f"  Engineered features found: {len(available_engineered)}")
    if available_engineered:
        print(f"    {', '.join(available_engineered)}")
    print(f"  Total selected features: {len(available_features)} (including {len(available_engineered)} engineered)")
    
    # Select features from both train and test
    train_df_selected = train_df[available_features].copy()
    test_df_selected = test_df[available_features].copy()
    
    return train_df_selected, test_df_selected

