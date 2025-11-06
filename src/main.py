"""
Main entry point for the Census Income Prediction pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loading import load_config, load_data, get_column_names
from utils import ensure_dir


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("Census Income Prediction Pipeline")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    
    # Load data
    print("\n2. Loading data...")
    train_df, test_df = load_data(
        config['data']['train_path'],
        config['data']['test_path'],
        config['data']['metadata_path']
    )
    
    # Set column names
    columns = get_column_names()
    train_df.columns = columns
    test_df.columns = columns
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Ensure output directories exist
    print("\n3. Setting up output directories...")
    ensure_dir(config['output']['models_dir'])
    ensure_dir(config['output']['results_dir'])
    ensure_dir(config['output']['figures_dir'])
    ensure_dir(config['output']['reports_dir'])
    ensure_dir(config['output']['predictions_dir'])
    
    print("\nPipeline setup complete!")
    print("\nNext steps:")
    print("  - Run exploratory data analysis: notebooks/01_exploratory_data_analysis.ipynb")
    print("  - Run data preprocessing: notebooks/02_data_preprocessing.ipynb")
    print("  - Run model experimentation: notebooks/03_model_experimentation.ipynb")


if __name__ == "__main__":
    main()

