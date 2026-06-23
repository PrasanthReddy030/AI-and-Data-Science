# House Prices -- Advanced Regression Techniques

**Kaggle Competition:** House Prices: Advanced Regression Techniques

Predict residential home sale prices in Ames, Iowa using 80 features covering
size, quality, location, age, and condition.

---

## Results

| Metric | Value |
|---|---|
| Validation RMSE (log scale) | 0.1231 |
| Validation R2 | 0.8985 |
| Validation MAPE | 8.39% |
| Baseline RMSE (DummyRegressor) | 0.3871 |
| Improvement over baseline | 68.2% |
| Best model | Ridge (tuned) |

---

## Workflow Architecture

```
Raw Data (train.csv / test.csv)
         |
         v
+-------------------------------------+
|  Sections 3-7: EDA                  |
|  Missing values . Distributions     |
|  Correlation . Outlier removal      |
+----------------|--------------------+
                 |
                 v
+-------------------------------------+
|  Section 8: Feature Engineering     |
|  +11 domain/interaction features    |
|  src/feature_engineering.py         |
+----------------|--------------------+
                 |
                 v
+-------------------------------------+
|  Sections 9-10: Pipeline            |
|  FeatureEngineer (custom)           |
|  ColumnTransformer                  |
|    numeric  Impute . log1p . Scale  |
|    ordinal  Impute . OrdEncode      |
|    nominal  Impute . Rare . OHE     |
|  src/transformers.py                |
+----------------|--------------------+
                 |
                 v
+-------------------------------------+
|  Sections 11-13: Modelling          |
|  Stratified 80/20 split             |
|  Benchmark 6 models (5-fold CV)     |
|  GridSearchCV tuning (3 models)     |
+----------------|--------------------+
                 |
                 v
+-------------------------------------+
|  Sections 14-18: Analysis           |
|  Residual diagnostics               |
|  Error analysis (price/nbhd)        |
|  Feature importance (3 methods)     |
|  SHAP explainability                |
|  Business insights                  |
+----------------|--------------------+
                 |
                 v
+-------------------------------------+
|  Section 19: Prediction             |
|  Retrain on full train set          |
|  Transform test set (no re-fit)     |
|  expm1() back to dollars            |
|  submission.csv                     |
+-------------------------------------+
```

---

## Model Comparison

| Model | CV RMSE | Notes |
|---|---|---|
| LinearRegression | ~0.160 | No regularisation; linear baseline |
| Ridge (tuned) | ~0.138 | L2 regularisation; stable |
| Lasso | ~0.158 | L1; automatic feature selection |
| ElasticNet | ~0.155 | L1+L2 combined |
| RandomForest (tuned) | ~0.136 | Ensemble; handles non-linearity |
| **Ridge (tuned)** | **~0.1231** | **Best model** |
| DummyRegressor (baseline) | ~0.390 | Predicts mean; floor reference |

---

## Top Features (SHAP consensus)

| Feature | Type | Why important |
|---|---|---|
| `OverallQual` | Raw | Strongest single predictor of price |
| `QualxSF` | Engineered | Quality x size interaction |
| `GrLivArea` / `TotalSF` | Raw / Engineered | Above-ground living area |
| `NeighborhoodMeanPrice` | Engineered | Location signal as continuous value |
| `GarageArea` / `GarageCars` | Raw | Garage near-essential in Ames, Iowa |
| `TotalBsmtSF` | Raw | Finished basement adds significant value |
| `YearBuilt` / `HouseAge` | Raw / Engineered | Newer homes command premium |

---

## Dataset

Data is sourced from the Kaggle competition and is not included in this repository due to Kaggle's terms of service.

**Download:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

After downloading, place the files in the `data/` directory:

```
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── data_description.txt
```

---

## Project Structure

```
House Price Predictor/
├── data/
│   ├── train.csv               # 1,460 rows x 81 cols (with SalePrice)  [not in repo]
│   ├── test.csv                # 1,459 rows x 80 cols (no SalePrice)    [not in repo]
│   ├── data_description.txt    # Feature descriptions
│   └── sample_submission.csv   # Kaggle submission format               [not in repo]
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py  # Domain feature functions
│   └── transformers.py         # Custom Scikit-Learn transformers
├── house_prices.ipynb          # Main notebook (20 sections)
├── submission.csv              # Final Kaggle submission
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Setup & Reproduction

```bash
cd "House Price Predictor"
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download data from Kaggle and place in data/ (see Dataset section above)

jupyter notebook house_prices.ipynb
```

All outputs are reproducible with `RANDOM_STATE = 42`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `log1p(SalePrice)` as target | Normalises right-skewed target; optimises RMSLE directly |
| Custom `FeatureEngineer` transformer | Keeps neighborhood stats inside CV -- prevents data leakage |
| `OrdinalEncoder` with domain order | Preserves monotonic quality signal (Ex > Gd > TA > Fa > Po) |
| `RareEncoder` before OHE | Prevents unseen-category errors on test set |
| Retrain on full data before test | More training data; val set used only for model selection |

---

## Business Insights

1. Moving from average (Q5) to good (Q7) overall quality adds ~$50-60k to median price.
2. Larger homes have lower $/sqft — adding space to a small home has higher marginal ROI.
3. Neighborhood alone spans a $200k+ median price range.
4. Remodelled homes command a ~$20-30k premium; a 2-car garage adds ~$30-40k vs no garage.
5. Model accuracy degrades outside the $150k-$350k range — predictions at extremes carry higher uncertainty.

---
