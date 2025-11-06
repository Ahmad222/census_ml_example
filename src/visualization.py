"""
Visualization utilities for Census Income prediction project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Any
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_target_distribution(
    y: pd.Series,
    title: str = "Target Variable Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    Plot target variable distribution.
    
    Parameters:
    -----------
    y : pd.Series
        Target variable
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    y.value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
    axes[0].set_title(f'{title} - Count')
    axes[0].set_xlabel('Income Level')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Pie chart
    y.value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    axes[1].set_title(f'{title} - Percentage')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_missing_values(
    df: pd.DataFrame,
    title: str = "Missing Values Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Plot missing values analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("No missing values found!")
        return
    
    plt.figure(figsize=(12, 6))
    missing.plot(kind='barh')
    plt.title(title)
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_cols: int = 3,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distributions of numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        List of columns to plot. If None, plots all numerical columns.
    n_cols : int
        Number of columns in subplot grid
    save_path : str, optional
        Path to save the figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if col in df.columns:
            df[col].hist(bins=50, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson',
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation matrix.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (numerical features only)
    method : str
        Correlation method: 'pearson', 'kendall', 'spearman'
    save_path : str, optional
        Path to save the figure
    """
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.empty:
        print("No numerical columns to plot correlation!")
        return
    
    corr = numerical_df.corr(method=method)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=False,
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f'Correlation Matrix ({method.capitalize()})')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance from tree-based models.
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : List[str]
        List of feature names
    top_n : int
        Number of top features to display
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute!")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot model comparison metrics.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame with model comparison metrics
    metrics : List[str]
        List of metrics to plot
    save_path : str, optional
        Path to save the figure
    """
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    if not available_metrics:
        print("No available metrics to plot!")
        return
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(available_metrics):
        comparison_df[metric].plot(kind='bar', ax=axes[idx], color='steelblue')
        axes[idx].set_title(f'{metric.capitalize()} Comparison')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

