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
│   ├── feature_engineering.py    # Feature engineering
│   ├── modeling.py               # Model training and evaluation
│   ├── visualization.py          # Visualization utilities
│   └── utils.py                  # Helper functions
│
├── notebooks/                     # Jupyter notebooks for EDA
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_experimentation.ipynb
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
   
   The project uses Jupyter notebooks for the complete workflow:
   - Start with: `notebooks/01_exploratory_data_analysis.ipynb`
   - Then: `notebooks/02_data_preprocessing.ipynb`
   - Finally: `notebooks/03_modeling.ipynb`

## Project Workflow

1. **Exploratory Data Analysis** (`notebooks/01_exploratory_data_analysis.ipynb`)
   - Data overview and statistics
   - Missing value analysis
   - Target variable distribution
   - Feature distributions and relationships

2. **Data Preprocessing** (`notebooks/02_data_preprocessing.ipynb`)
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features
   - Train/test split

3. **Modeling** (`notebooks/03_model_experimentation.ipynb`)
   - Train models (LightGBM, Random Forest)
   - Hyperparameter tuning
   - Model evaluation and comparison
   - Feature importance analysis

4. **Final Model Selection**
   - Select best model based on performance metrics
   - Generate predictions on test set
   - Create evaluation report

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
