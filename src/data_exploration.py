"""
Data exploration utilities for Census Income dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def analyze_missing_values(
    df: pd.DataFrame,
    missing_indicators: List[str] = ["?", " ?", "? ", " ? "]
) -> pd.DataFrame:
    """
    Analyze missing values including special indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    missing_indicators : List[str]
        List of strings that indicate missing values
        
    Returns:
    --------
    missing_df : pd.DataFrame
        DataFrame with missing value statistics
    """
    missing_stats = []
    
    for col in df.columns:
        # Count NaN
        nan_count = df[col].isna().sum()
        
        # Count missing indicators
        indicator_count = 0
        for indicator in missing_indicators:
            indicator_count += (df[col].astype(str) == indicator).sum()
        
        total_missing = nan_count + indicator_count
        missing_pct = (total_missing / len(df)) * 100
        
        missing_stats.append({
            'column': col,
            'missing_count': total_missing,
            'missing_percentage': missing_pct,
            'data_type': df[col].dtype,
            'unique_values': df[col].nunique()
        })
    
    missing_df = pd.DataFrame(missing_stats)
    missing_df = missing_df.sort_values('missing_percentage', ascending=False)
    
    return missing_df


def analyze_duplicate_values(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze duplicate rows in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    subset : List[str], optional
        List of column names to consider for duplicate detection.
        If None, considers all columns.
        
    Returns:
    --------
    duplicate_summary : pd.DataFrame
        Summary statistics about duplicates
    duplicate_rows : pd.DataFrame
        DataFrame containing duplicate rows (if any)
    """
    # Check for completely duplicate rows
    duplicate_mask = df.duplicated(keep=False)
    total_duplicates = duplicate_mask.sum()
    unique_duplicate_groups = df.duplicated(keep='first').sum()
    
    # Check for duplicates in subset of columns
    if subset:
        subset_duplicate_mask = df.duplicated(subset=subset, keep=False)
        subset_total_duplicates = subset_duplicate_mask.sum()
        subset_unique_groups = df.duplicated(subset=subset, keep='first').sum()
    else:
        subset_total_duplicates = 0
        subset_unique_groups = 0
    
    # Get duplicate rows
    if total_duplicates > 0:
        duplicate_rows = df[duplicate_mask].sort_values(by=df.columns.tolist())
    else:
        duplicate_rows = pd.DataFrame()
    
    # Create summary
    summary = {
        'metric': [
            'Total duplicate rows (all columns)',
            'Unique duplicate groups (all columns)',
            'Percentage of duplicates (all columns)',
            'Total duplicate rows (subset)' if subset else 'N/A',
            'Unique duplicate groups (subset)' if subset else 'N/A',
            'Percentage of duplicates (subset)' if subset else 'N/A'
        ],
        'value': [
            total_duplicates,
            unique_duplicate_groups,
            (total_duplicates / len(df)) * 100 if len(df) > 0 else 0,
            subset_total_duplicates if subset else 'N/A',
            subset_unique_groups if subset else 'N/A',
            (subset_total_duplicates / len(df)) * 100 if subset and len(df) > 0 else 'N/A'
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    
    return summary_df, duplicate_rows


def plot_duplicate_analysis(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize duplicate value analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    subset : List[str], optional
        List of column names to consider for duplicate detection
    save_path : str, optional
        Path to save the figure
    """
    duplicate_mask = df.duplicated(keep=False)
    total_duplicates = duplicate_mask.sum()
    total_rows = len(df)
    unique_rows = total_rows - total_duplicates + df.duplicated(keep='first').sum()
    
    if subset:
        subset_duplicate_mask = df.duplicated(subset=subset, keep=False)
        subset_total_duplicates = subset_duplicate_mask.sum()
    else:
        subset_total_duplicates = 0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Duplicate vs Unique rows
    if total_duplicates > 0:
        duplicate_counts = [unique_rows, total_duplicates]
        labels = ['Unique Rows', 'Duplicate Rows']
        colors = ['lightblue', 'coral']
        
        axes[0].pie(duplicate_counts, labels=labels, autopct='%1.1f%%', 
                   colors=colors, startangle=90, 
                   wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        axes[0].set_title('Duplicate Rows Distribution (All Columns)')
    else:
        axes[0].text(0.5, 0.5, 'No Duplicates Found', 
                    ha='center', va='center', fontsize=14, weight='bold')
        axes[0].set_title('Duplicate Rows Distribution (All Columns)')
    
    # Plot 2: Duplicate analysis by column (if subset provided)
    if subset and subset_total_duplicates > 0:
        # Count duplicates per column in subset
        col_duplicate_counts = []
        col_names = []
        for col in subset:
            if col in df.columns:
                col_dup = df.duplicated(subset=[col], keep=False).sum()
                col_duplicate_counts.append(col_dup)
                col_names.append(col)
        
        if col_duplicate_counts:
            axes[1].barh(col_names, col_duplicate_counts, color='steelblue', edgecolor='black')
            axes[1].set_xlabel('Number of Duplicate Rows')
            axes[1].set_title('Duplicates by Column (Subset)')
            axes[1].grid(axis='x', alpha=0.3)
    else:
        # Show overall duplicate statistics
        stats_text = f"Total Rows: {total_rows:,}\n"
        stats_text += f"Unique Rows: {unique_rows:,}\n"
        stats_text += f"Duplicate Rows: {total_duplicates:,}\n"
        stats_text += f"Duplicate %: {(total_duplicates/total_rows)*100:.2f}%"
        
        axes[1].text(0.5, 0.5, stats_text, ha='center', va='center', 
                    fontsize=12, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].set_title('Duplicate Statistics')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of categorical feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    top_n : int, optional
        Number of top categories to show
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    value_counts = df[column].value_counts()
    
    if top_n:
        value_counts = value_counts.head(top_n)
    
    plt.figure(figsize=(12, 6))
    value_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_numerical_vs_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
    use_log_scale: Optional[bool] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot numerical feature distribution by target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature : str
        Numerical feature name
    target : str
        Target variable name
    use_log_scale : bool, optional
        Whether to use logarithmic scale. If None, automatically detects based on data range.
    save_path : str, optional
        Path to save the figure
    """
    # Convert to numeric, handling any non-numeric values
    numeric_data = pd.to_numeric(df[feature], errors='coerce')
    
    # Determine if log scale should be used
    if use_log_scale is None:
        # Auto-detect: use log scale if range is large (max/min > 100) and all values >= 0
        min_val = numeric_data.min()
        max_val = numeric_data.max()
        if min_val > 0 and max_val > 0:
            range_ratio = max_val / min_val if min_val > 0 else float('inf')
            use_log_scale = range_ratio > 100
        else:
            # If there are zeros or negatives, use log scale only if explicitly requested
            use_log_scale = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data for plotting
    plot_data = df.copy()
    plot_data[feature] = numeric_data
    
    # Box plot
    if use_log_scale:
        # For log scale, filter out zeros and negatives
        plot_data_log = plot_data[plot_data[feature] > 0].copy()
        if len(plot_data_log) > 0:
            plot_data_log.boxplot(column=feature, by=target, ax=axes[0])
            axes[0].set_yscale('log')
            axes[0].set_ylabel(f'{feature} (log scale)')
        else:
            # If no positive values, use regular scale
            plot_data.boxplot(column=feature, by=target, ax=axes[0])
            axes[0].set_ylabel(feature)
    else:
        plot_data.boxplot(column=feature, by=target, ax=axes[0])
        axes[0].set_ylabel(feature)
    
    axes[0].set_title('')
    axes[0].set_xlabel(target)
    axes[0].grid(alpha=0.3)
    
    # Histogram
    for target_value in df[target].unique():
        subset = plot_data[plot_data[target] == target_value][feature].dropna()
        
        if use_log_scale:
            # Filter out zeros and negatives for log scale
            subset = subset[subset > 0]
            if len(subset) > 0:
                axes[1].hist(subset, alpha=0.6, label=str(target_value), bins=50)
        else:
            axes[1].hist(subset, alpha=0.6, label=str(target_value), bins=50)
    
    if use_log_scale and len(plot_data[plot_data[feature] > 0]) > 0:
        axes[1].set_yscale('log')
        axes[1].set_xscale('log')
        axes[1].set_xlabel(f'{feature} (log scale)')
        axes[1].set_ylabel('Frequency (log scale)')
    else:
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Frequency')
    
    axes[1].set_title(f'{feature} Distribution by {target}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    scale_note = " (log scale)" if use_log_scale else ""
    plt.suptitle(f'{feature} vs {target}{scale_note}', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_categorical_vs_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
    top_n: Optional[int] = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot categorical feature distribution by target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature : str
        Categorical feature name
    target : str
        Target variable name
    top_n : int, optional
        Number of top categories to show
    save_path : str, optional
        Path to save the figure
    """
    # Create crosstab
    crosstab = pd.crosstab(df[feature], df[target])
    
    if top_n:
        # Get top N categories by total count
        top_categories = df[feature].value_counts().head(top_n).index
        crosstab = crosstab.loc[top_categories]
    
    # Calculate percentages
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count plot
    crosstab.plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'], edgecolor='black')
    axes[0].set_title(f'{feature} vs {target} - Count')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Count')
    axes[0].legend(title=target)
    axes[0].tick_params(axis='x', rotation=45)
    plt.setp(axes[0].xaxis.get_majorticklabels(), ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Percentage plot
    crosstab_pct.plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'], edgecolor='black')
    axes[1].set_title(f'{feature} vs {target} - Percentage')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Percentage (%)')
    axes[1].legend(title=target)
    axes[1].tick_params(axis='x', rotation=45)
    plt.setp(axes[1].xaxis.get_majorticklabels(), ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_correlation_with_target(
    df: pd.DataFrame,
    target: str,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation of numerical features with target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Target variable name
    top_n : int
        Number of top features to show
    save_path : str, optional
        Path to save the figure
    """
    # Encode target if categorical
    df_encoded = df.copy()
    if df[target].dtype == 'object':
        df_encoded[target] = pd.Categorical(df[target]).codes
    
    # Calculate correlations
    numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    if target in numerical_cols:
        numerical_cols.remove(target)
    
    correlations = df_encoded[numerical_cols + [target]].corr()[target].sort_values(ascending=False)
    correlations = correlations.drop(target)
    correlations = correlations.head(top_n)
    
    plt.figure(figsize=(10, max(8, len(correlations) * 0.4)))
    correlations.plot(kind='barh', color='steelblue', edgecolor='black')
    plt.title(f'Top {top_n} Features Correlated with {target}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive data summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary statistics
    """
    summary = []
    
    for col in df.columns:
        col_info = {
            'column': col,
            'dtype': str(df[col].dtype),
            'non_null_count': df[col].notna().sum(),
            'null_count': df[col].isna().sum(),
            'null_percentage': (df[col].isna().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'unique_percentage': (df[col].nunique() / len(df)) * 100
        }
        
        # Check if column is numeric using pandas API
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        if is_numeric:
            # Convert to numeric, handling any non-numeric values
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            col_info.update({
                'mean': numeric_series.mean(),
                'median': numeric_series.median(),
                'std': numeric_series.std(),
                'min': numeric_series.min(),
                'max': numeric_series.max(),
                'q25': numeric_series.quantile(0.25),
                'q75': numeric_series.quantile(0.75)
            })
        else:
            col_info.update({
                'most_frequent': df[col].mode()[0] if not df[col].mode().empty else None,
                'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            })
        
        summary.append(col_info)
    
    return pd.DataFrame(summary)


def plot_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    save_path: Optional[str] = None
) -> None:
    """
    Plot box plots to identify outliers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        List of columns to plot. If None, plots all numerical columns.
    method : str
        Method for outlier detection: 'iqr' or 'zscore'
    save_path : str, optional
        Path to save the figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if len(columns) > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if col in df.columns:
            df.boxplot(column=col, ax=axes[idx], vert=True)
            axes[idx].set_title(f'Outliers in {col}')
            axes[idx].grid(alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance_by_target(
    df: pd.DataFrame,
    categorical_features: List[str],
    target: str,
    top_n: int = 15,
    save_path: Optional[str] = None
) -> None:
    """
    Calculate and plot feature importance for categorical features using chi-square like analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_features : List[str]
        List of categorical feature names
    target : str
        Target variable name
    top_n : int
        Number of top features to show
    save_path : str, optional
        Path to save the figure
    """
    from scipy.stats import chi2_contingency
    
    importance_scores = []
    
    for feature in categorical_features:
        if feature in df.columns and feature != target:
            try:
                # Create contingency table
                contingency = pd.crosstab(df[feature], df[target])
                
                # Calculate chi-square statistic
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                # Calculate Cramér's V (effect size)
                n = contingency.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                
                importance_scores.append({
                    'feature': feature,
                    'chi2': chi2,
                    'p_value': p_value,
                    'cramers_v': cramers_v
                })
            except:
                continue
    
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('cramers_v', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, max(8, len(importance_df) * 0.4)))
    plt.barh(range(len(importance_df)), importance_df['cramers_v'], color='steelblue', edgecolor='black')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel("Cramér's V (Effect Size)")
    plt.title(f'Top {top_n} Categorical Features by Association with {target}')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return importance_df

