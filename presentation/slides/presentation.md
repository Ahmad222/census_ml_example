# Census Income Prediction
## Machine Learning Project

**Dataiku Technical Assessment**  
**November 2025**

---

## Problem Formulation

### Objective
**Predict if income > $50,000/year**

📊 **Dataset:**
- ~300K individuals
- 40+ demographic & employment features
- Binary classification task

![Target Distribution](../../results/figures/target_distribution.png)

---

## Dataset Overview

📈 **Scale:**
- Training: ~200K samples
- Test: ~100K samples
- 41 features (13 numerical, 29 categorical)

⚠️ **Challenge:**
- **Severe class imbalance** (6.21% positive class)
- Requires specialized approaches

![Target Distribution](../../results/figures/target_distribution.png)

---

## Exploratory Data Analysis

### Data Quality Assessment

✅ **Clean dataset** (~200K samples)  
✅ **Missing values** well-documented  
⚠️ **Outliers** in financial features

![Missing Values](../../results/figures/missing_values.png)

---

## Exploratory Data Analysis

### Strong Predictors Identified

🎯 **Top Features:**
- Education level
- Occupation & Industry  
- Weeks worked
- Age
- Marital status

![Education vs Target](../../results/figures/education_vs_target.png)

---

## Exploratory Data Analysis

### Feature Importance

📊 **Categorical Features:**
- Education, Occupation, Industry → Strongest associations
- Marital status, Class of worker → Highly predictive

![Categorical Feature Importance](../../results/figures/categorical_feature_importance.png)

---

## Exploratory Data Analysis

### Numerical Features

📈 **Strong Correlations:**
- Weeks worked → Strong positive
- Age → Moderate positive
- Financial features → Highly predictive when non-zero

![Feature Correlation](../../results/figures/feature_correlation_target.png)

---

## Approach: Data Preprocessing

### Preprocessing Steps

1. **Remove Duplicates** (54K duplicates)
2. **Handle Missing Values** (categorical: "not identified", numerical: median)
3. **Treat Outliers** (winsorization)
4. **Feature Engineering** (6 new features)
5. **Feature Selection** (top 30 + engineered)
6. **Encode Categoricals** (hybrid: one-hot/frequency)

![Outliers](../../results/figures/outliers.png)

---

## Approach: Modeling

### Models
- **LightGBM** (primary) - Optuna tuning
- **Random Forest** (alternative) - GridSearchCV

### Hyperparameter Tuning
- **70 trials** (50 random + 20 TPE)
- Bayesian optimization
- Optimize for **ROC-AUC**

### Class Imbalance
- Class weights applied
- ROC-AUC as primary metric

---

## Approach: Evaluation

### Strategy
- **Train/Val/Test** split
- **ROC-AUC** primary metric
- **Comprehensive visualization**

### Metrics
- ROC-AUC, Precision, Recall, F1
- ROC & PR curves
- Confusion matrix

---

## Results: Model Performance

### Performance Dashboard

![Model Performance](../../results/figures/model_performance.png)

**Metrics:**
- ROC-AUC (primary)
- Precision-Recall curves
- Confusion matrix

---

## Results: Hyperparameter Optimization

### Optuna Tuning Progress

**70 trials** → Best hyperparameters selected

![Optuna Optimization](../../results/figures/optuna_optimization_history.png)

---

## Results: Key Insights

✅ **Good generalization** (small train-val gap)  
✅ **Strong discrimination** (high ROC-AUC)  
✅ **Effective imbalance handling**

**Top Predictors:**
- Education, Occupation
- Financial features (when non-zero)
- Weeks worked, Age

---

## Next Steps

### High Priority

1. **Ensemble Methods**  
   Combine LightGBM + Random Forest

2. **Feature Engineering**  
   Interaction features (age × education)

3. **Threshold Optimization**  
   Cost-sensitive learning

4. **Model Interpretability**  
   SHAP values, feature explanations

---

## Next Steps

### Medium Priority

5. **SMOTE** - Synthetic oversampling  
6. **Additional Models** - CatBoost, XGBoost  
7. **Advanced Feature Selection** - RFE, SelectFromModel  
8. **Nested CV** - More robust tuning

---

## Summary

### What We Built
✅ **Comprehensive EDA** - Feature importance analysis  
✅ **Robust Preprocessing** - Production-ready pipeline  
✅ **Optimized Models** - Hyperparameter tuning  
✅ **Thorough Evaluation** - Multiple metrics & visualizations

### Key Achievements
- Handled **severe class imbalance** (6.21%)
- Identified **strong predictors**
- Built **reproducible pipeline**
- Achieved **good model performance**

### Impact
🚀 Ready for deployment  
📈 Clear improvement path  
📚 Well-documented

---

## Thank You

**Questions?**

**Project Repository:** [GitHub Link]  
**Reports:** `results/reports/`  
**Notebooks:** `notebooks/`

