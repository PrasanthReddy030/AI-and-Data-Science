# Titanic Survival Predictor

## Problem Statement

Predict whether a passenger survived the Titanic disaster using demographic and ticket data. Binary classification (survived = 1 / 0) based on the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).

## Project Structure

```
Titanic Survival Predictor/
├── Titanic_Survival_Predictor.ipynb   # Main notebook
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv                      # Training data (891 rows)
│   ├── test.csv                       # Kaggle test set (labels withheld)
│   ├── gender_submission.csv          # Kaggle sample submission
│   └── titanic.zip                    # Original archive
└── assets/
    ├── eda_survival_sex.png
    ├── eda_survival_class.png
    └── model_comparison.png
```

## Dataset

**Source:** Kaggle Titanic — Machine Learning from Disaster (`data/train.csv`)

| Attribute | Detail |
|-----------|--------|
| Rows | 891 passengers |
| Features | Age, Fare, SibSp, Parch, Sex, Pclass, Embarked |
| Target | `Survived` (0 / 1) |
| Missing values | Age (~177), Cabin (~687, dropped), Embarked (2) |

**EDA highlights:**
- Women survived at ~74% vs men at ~19%
- 1st-class passengers survived at ~63% vs 3rd-class at ~24%

## Approach

### Preprocessing

| Feature type | Strategy |
|---|---|
| Numeric (`Age`, `Fare`, `SibSp`, `Parch`) | Median imputation → standard scaling |
| Categorical (`Sex`, `Pclass`, `Embarked`) | Mode imputation → one-hot encoding |

- `Cabin` dropped (>77% missing — imputing would add noise, not signal)
- Stratified 80/20 split preserves the ~38% survival rate across train and test

### Models

All four classifiers share the same preprocessing pipeline:

| Model | Key Hyperparameters |
|-------|---------------------|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=5` |
| SVM (RBF) | `kernel='rbf'`, `probability=True` |
| KNN | `n_neighbors=7` |

## Results

| Model | F1 | ROC-AUC |
|---|---|---|
| **SVM** | **0.736** | **0.846** |
| Logistic Regression | 0.724 | 0.843 |
| KNN | 0.709 | 0.846 |
| Decision Tree | 0.656 | 0.797 |

*Test set: 179 passengers (110 non-survivors, 69 survivors). Sorted by F1.*

## Key Findings

- **SVM is the best overall model** — highest F1 (0.736) and tied ROC-AUC (0.846). It edges KNN on hard predictions despite identical ranking scores.
- **Logistic Regression is the practical baseline** — within 0.003 ROC-AUC of SVM with no tuning; preferred when explainability matters.
- **Decision Tree underperforms** — lags ~5 points on ROC-AUC (0.797) even with depth capping, suggesting the boundary benefits from smooth non-linearity.
- **Survivor recall is the shared weakness** — class 1 recall ranges 0.58–0.67 vs 0.88–0.91 for class 0, reflecting the 110:69 imbalance. Class weighting or threshold tuning would help.
- **Feature engineering headroom remains** — `Name` (title extraction), `Cabin` (deck letter), and `Sex × Pclass` interactions are untouched and likely the next performance lever.
