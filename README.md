# Census Income Prediction - Dataiku Technical Assessment

## Project Overview

This project aims to predict whether a person makes more or less than $50,000 per year using US Census data. The dataset contains ~300,000 individuals with 40 attributes (7 continuous, 33 nominal).

## Repository Structure

```
census_ml_example/
├── data/                          # Raw data files (do not modify)
│   ├── census_income_learn.csv
│   ├── census_income_test.csv
│   ├── census_income_metadata.txt
│   └── census_income_additional_info.pdf
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_loading.py           # Data loading utilities
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── data_exploration.py       # EDA utilities and visualizations
│   ├── modeling.py               # Model training and evaluation
│   └── visualization.py          # Visualization utilities
│
├── notebooks/                     # Jupyter notebooks (run in order)
│   ├── 01_exploratory_data_analysis.ipynb    # EDA and feature importance
│   ├── 02_data_preprocessing.ipynb           # Data cleaning and preprocessing
│   └── 03_modeling.ipynb                     # Model training and evaluation
│
├── config/                        # Configuration files
│   ├── config.yaml               # Main configuration
│   └── model_config.yaml         # Model-specific configurations
│
├── models/                        # Trained models (gitignored)
│   └── .gitkeep
│
├── results/                       # Results and outputs
│   ├── figures/                  # Generated plots and visualizations
│   ├── reports/                  # Model evaluation reports
│   └── predictions/              # Prediction outputs
│
├── presentation/                  # Presentation materials
│   ├── slides/                   # Presentation slides
│   └── notes/                    # Presentation notes
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_preprocessing.py
│
├── pyproject.toml                 # Project configuration and dependencies (uv)
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Setup Instructions

1. **Clone the repository** (if applicable)
   ```bash
   git clone <repository-url>
   cd census_ml_example
   ```

2. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or using pip: pip install uv
   ```

3. **Create virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

   Or use uv directly without activating venv:
   ```bash
   uv sync
   ```

4. **Run the analysis pipeline**
   
   The project uses Jupyter notebooks for the complete workflow. Execute them in order:
   
   ```bash
   # Start Jupyter (if not already running)
   jupyter notebook
   # Or: jupyter lab
   ```
   
   Then run the notebooks in sequence:
   1. **Exploratory Data Analysis:** `notebooks/01_exploratory_data_analysis.ipynb`
      - Analyzes data quality, distributions, correlations, and feature importance
      - Generates EDA figures and saves feature importance for preprocessing
   
   2. **Data Preprocessing:** `notebooks/02_data_preprocessing.ipynb`
      - Removes duplicates, handles missing values, treats outliers
      - Performs feature engineering and feature selection
      - Encodes categoricals and saves processed data
   
   3. **Modeling:** `notebooks/03_modeling.ipynb`
      - Trains models (LightGBM or Random Forest)
      - Performs hyperparameter tuning
      - Evaluates and visualizes model performance

## Project Workflow

The project follows a sequential notebook-based workflow:

### 1. Exploratory Data Analysis (`notebooks/01_exploratory_data_analysis.ipynb`)
- Data overview and statistics
- Missing value analysis
- Target variable distribution (class imbalance analysis)
- Feature distributions (numerical and categorical)
- Correlation analysis
- Feature importance calculation (for feature selection)
- Outlier detection
- **Output:** EDA figures and feature importance CSV file

### 2. Data Preprocessing (`notebooks/02_data_preprocessing.ipynb`)
- Remove duplicate rows
- Handle missing values (categorical: "not identified", numerical: median)
- Treat outliers (winsorization with train-derived bounds)
- Convert target to binary (0/1)
- Feature engineering (binary flags, aggregations, interactions)
- Feature selection (based on EDA importance + engineered features)
- Encode categoricals (hybrid: one-hot for ≤5 categories, frequency for >5)
- **Output:** Processed data files (raw and encoded versions)

### 3. Modeling (`notebooks/03_modeling.ipynb`)
- Model selection (LightGBM or Random Forest, configured in `config/config.yaml`)
- Load preprocessed data
- Hyperparameter tuning:
  - LightGBM: Optuna (random search + TPE)
  - Random Forest: GridSearchCV
- Train best model with optimized hyperparameters
- Evaluate on train, validation, and test sets
- Visualize performance (metrics, ROC curves, PR curves, confusion matrix)
- Save model and hyperparameters
- **Output:** Trained model, performance metrics, and visualization figures

### 4. Reports
- **EDA Report:** `results/reports/eda_report.md`
- **Preprocessing Report:** `results/reports/preprocessing_report.md`
- **Modeling Report:** `results/reports/modeling_report.md`

## Key Findings

(To be updated after analysis)

## Model Performance

(To be updated after analysis)

## Future Improvements

(To be updated after analysis)

## Author

[Your Name]

## License

This project is for assessment purposes only.
