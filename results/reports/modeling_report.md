# Modeling Report
## Census Income Prediction Project

**Date:** 2025-11-07  
**Dataset:** Census Income (Adult) Dataset  
**Analysis Type:** Model Training and Evaluation

---

## Executive Summary

This report documents the machine learning modeling process for predicting whether an individual's income exceeds $50,000 per year. The project implements two models: **LightGBM** and **Random Forest**, with comprehensive hyperparameter tuning and evaluation.

**Key Highlights:**
- Model selection via configuration (`config/config.yaml`)
- Hyperparameter tuning using Optuna (LightGBM) and GridSearchCV (Random Forest)
- Class imbalance handling with class weights
- Comprehensive evaluation across train, validation, and test sets
- Performance visualization with multiple metrics

---

## 1. Model Selection

**Configuration-Based Selection:**
- Model type is configured in `config/config.yaml` under `modeling.model_type`
- Supported models: `lightgbm`, `random_forest`
- Selection determines:
  - Data version (raw vs processed)
  - Hyperparameter tuning method
  - Model-specific configurations

**Current Configuration:**
- **Selected Model:** LightGBM (default)
- **Data Version:** Processed (encoded categorical features)
- **Hyperparameter Tuning:** Optuna (Bayesian optimization)

---

## 2. Data Preparation

**Input Data:**
- Training data: `data/processed/train_full_processed.csv` (combined train+val)
- Test data: `data/processed/test_processed.csv`

**Data Splitting:**
- Combined train+val data is split for hyperparameter tuning
- Split ratio: 80% train, 20% validation
- Stratified split to maintain class distribution
- Test set kept separate for final evaluation

**Feature Set:**
- Selected features based on EDA importance (top 30 features)
- Includes all engineered features (6 features)
- Total features: ~30-36 (depending on feature selection configuration)

---

## 3. Hyperparameter Tuning

### 3.1 LightGBM (Optuna)

**Optimization Strategy:**
- **Random Search:** Initial exploration (default: 50 trials)
- **TPE (Tree-structured Parzen Estimator):** Refined optimization (default: 20 trials)
- **Total Trials:** 70 trials
- **Optimization Metric:** ROC-AUC (handles class imbalance)

**Hyperparameter Ranges:**
- `n_estimators`: 100-1000
- `max_depth`: 3-8
- `learning_rate`: 0.001-0.3 (log scale)
- `num_leaves`: 15-63
- `feature_fraction`: 0.4-0.8
- `bagging_fraction`: 0.4-0.8
- `min_child_samples`: 20-200
- `reg_alpha`: 0.0001-50.0 (log scale)
- `reg_lambda`: 0.0001-50.0 (log scale)
- `min_gain_to_split`: 0.0-1.0

**Class Imbalance Handling:**
- `scale_pos_weight` automatically calculated based on class distribution
- Formula: `num_negative_samples / num_positive_samples`

**Visualization:**
- Optimization history plot showing trial performance over time
- Saved to: `results/figures/optuna_optimization_history.png`

![Optuna Optimization History](../figures/optuna_optimization_history.png)

### 3.2 Random Forest (GridSearchCV)

**Optimization Strategy:**
- **GridSearchCV:** Exhaustive search over parameter grid
- **Cross-Validation:** 5-fold stratified CV
- **Optimization Metric:** ROC-AUC

**Hyperparameter Grid:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ["sqrt", "log2"]

**Class Imbalance Handling:**
- `class_weight='balanced'` to automatically adjust for class imbalance

---

## 4. Model Training

**Training Process:**
1. Load best hyperparameters from tuning
2. Train final model on full training data (train+val combined)
3. Use best hyperparameters from validation performance
4. Apply class weights for imbalanced data

**Model Configuration:**
- **LightGBM:** Uses native categorical feature support (if raw data) or processed features
- **Random Forest:** Uses processed (encoded) features
- Both models configured with class weights to handle imbalance

---

## 5. Model Evaluation

### 5.1 Evaluation Metrics

**Primary Metrics:**
- **ROC-AUC:** Area Under ROC Curve (primary metric for imbalanced data)
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall

**Additional Metrics:**
- **Confusion Matrix:** Detailed breakdown of predictions
- **ROC Curve:** True Positive Rate vs False Positive Rate
- **Precision-Recall Curve:** Precision vs Recall trade-off

### 5.2 Evaluation Strategy

**Three-Way Evaluation:**
1. **Training Set:** Assess model fit and potential overfitting
2. **Validation Set:** Assess generalization (used for hyperparameter selection)
3. **Test Set:** Final unbiased evaluation (held out until final assessment)

**Key Considerations:**
- Monitor train vs validation performance gap (overfitting detection)
- Focus on ROC-AUC as primary metric (robust to class imbalance)
- Consider Precision-Recall curve for imbalanced classification

---

## 6. Performance Visualization

**Comprehensive Performance Dashboard:**
- Saved to: `results/figures/model_performance.png`

The visualization includes four subplots:

### 6.1 Metrics Comparison Bar Chart
- Compares Precision, Recall, F1-Score, and ROC-AUC
- Shows performance across Train, Validation, and Test sets
- Helps identify overfitting (large train-val gap)

![Model Performance](../figures/model_performance.png)

### 6.2 ROC Curves
- Shows True Positive Rate vs False Positive Rate
- Displays AUC score for each dataset
- Includes random classifier baseline (diagonal line)
- Higher AUC indicates better discrimination ability

### 6.3 Precision-Recall Curves
- Shows Precision vs Recall trade-off
- Important for imbalanced classification
- Better than ROC for highly imbalanced datasets
- Higher area under PR curve indicates better performance

### 6.4 Confusion Matrix (Test Set)
- Detailed breakdown of predictions on test set
- Shows:
  - True Negatives (TN): Correctly predicted ≤$50K
  - False Positives (FP): Incorrectly predicted >$50K
  - False Negatives (FN): Incorrectly predicted ≤$50K
  - True Positives (TP): Correctly predicted >$50K

---

## 7. Model Performance Summary

### 7.1 Performance Metrics

**Example Results (to be updated with actual values):**

| Dataset | Precision | Recall | F1-Score | ROC-AUC |
|---------|-----------|--------|----------|---------|
| Train   | X.XXX     | X.XXX  | X.XXX    | X.XXX   |
| Val     | X.XXX     | X.XXX  | X.XXX    | X.XXX   |
| Test    | X.XXX     | X.XXX  | X.XXX    | X.XXX   |

### 7.2 Overfitting Analysis

**Indicators:**
- **Train-Val Gap:** Difference between train and validation performance
- **Large Gap:** Indicates overfitting (model memorizing training data)
- **Small Gap:** Indicates good generalization

**Mitigation Strategies (if overfitting detected):**
- Increase regularization (`reg_alpha`, `reg_lambda`)
- Reduce model complexity (`max_depth`, `num_leaves`)
- Increase minimum samples per leaf (`min_child_samples`)
- Reduce feature fraction (`feature_fraction`)
- Increase bagging fraction variability

### 7.3 Class Imbalance Impact

**Challenges:**
- Only 6.21% of samples in positive class (>$50K)
- Model may bias toward majority class
- Accuracy can be misleading (high accuracy with all-negative predictions)

**Solutions Applied:**
- Class weights (`scale_pos_weight` for LightGBM, `class_weight='balanced'` for RF)
- ROC-AUC as primary metric (threshold-independent)
- Precision-Recall curves for threshold selection

---

## 8. Model Saving

**Saved Artifacts:**
- **Trained Model:** `models/best_model.pkl` or `models/best_model.txt` (LightGBM)
- **Best Hyperparameters:** `models/best_hyperparameters.json`
- **Selected Features:** `models/selected_features.json` (if feature selection enabled)

**Model Persistence:**
- LightGBM: Saved using native `save_model()` method
- Random Forest: Saved using `joblib.dump()`
- Hyperparameters: Saved as JSON for reproducibility

---

## 9. Key Findings and Insights

### 9.1 Model Performance
- (To be updated with actual results)
- Model shows good generalization if train-val gap is small
- ROC-AUC indicates strong discrimination ability

### 9.2 Hyperparameter Insights
- (To be updated with best hyperparameters found)
- Optimal complexity level identified through tuning
- Regularization helps prevent overfitting

### 9.3 Feature Importance
- (To be updated if feature importance analysis is added)
- Top features contributing to predictions
- Feature selection impact on performance

---

## 10. Recommendations

### 10.1 Model Selection
- **LightGBM:** Recommended for:
  - Fast training and prediction
  - Native categorical feature support
  - Good performance on imbalanced data
  - Efficient hyperparameter tuning with Optuna

- **Random Forest:** Alternative for:
  - Interpretability (feature importance)
  - Robust to overfitting
  - No hyperparameter tuning complexity

### 10.2 Hyperparameter Tuning
- **Optuna (LightGBM):** Efficient Bayesian optimization
  - Start with random search for exploration
  - Use TPE for refinement
  - Monitor optimization history for convergence

- **GridSearchCV (Random Forest):** Exhaustive search
  - Smaller parameter grids due to computational cost
  - Use RandomizedSearchCV for larger grids

### 10.3 Evaluation Strategy
- **Primary Metric:** ROC-AUC (threshold-independent, handles imbalance)
- **Secondary Metrics:** Precision, Recall, F1-Score
- **Visualization:** ROC and PR curves for comprehensive assessment
- **Overfitting Detection:** Monitor train-val performance gap

### 10.4 Future Improvements
1. **Ensemble Methods:** Combine LightGBM and Random Forest predictions
2. **Threshold Optimization:** Tune classification threshold based on business costs
3. **Feature Engineering:** Explore additional feature interactions
4. **Cross-Validation:** Use k-fold CV for more robust hyperparameter selection
5. **Model Interpretability:** Add SHAP values or feature importance analysis

---

## 11. Configuration Files

**Model Configuration:** `config/model_config.yaml`
- Hyperparameter ranges for Optuna
- Model-specific settings

**Main Configuration:** `config/config.yaml`
- Model selection (`modeling.model_type`)
- Hyperparameter tuning settings (`modeling.hpt`)
- Evaluation metrics (`modeling.scoring_metric`)

---

## Appendix: Generated Figures

All model performance figures are saved in `results/figures/`:

- `optuna_optimization_history.png` - Optuna optimization history (LightGBM only)
- `model_performance.png` - Comprehensive performance dashboard:
  - Metrics comparison bar chart
  - ROC curves (Train, Val, Test)
  - Precision-Recall curves (Train, Val, Test)
  - Confusion matrix (Test set)

---

**Report Generated:** 2025-01-XX  
**Modeling Notebook:** `notebooks/03_modeling.ipynb`  
**Preprocessing Report Reference:** `results/reports/preprocessing_report.md`  
**EDA Report Reference:** `results/reports/eda_report.md`

