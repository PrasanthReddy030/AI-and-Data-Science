"""
Custom Scikit-Learn transformers for the Ames Housing dataset.

Each transformer follows the BaseEstimator + TransformerMixin protocol:
  - fit(X, y)      : learn statistics from training data
  - transform(X)   : apply transformation (no re-learning)
  - fit_transform() : provided free by TransformerMixin

Usage in a Pipeline:
    from src.transformers import FeatureEngineer, RareEncoder, SkewnessTransformer
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import src.feature_engineering as fe


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Apply all domain feature engineering steps.

    Learns neighborhood mean price from training targets at fit() time,
    then applies a consistent mapping at transform() time -- preventing leakage.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if y is not None:
            y_series = pd.Series(y, index=X.index, name="SalePrice")
            tmp = X.copy()
            tmp["SalePrice"] = y_series
            self.nbhd_stats_ = tmp.groupby("Neighborhood")["SalePrice"].mean()
        else:
            self.nbhd_stats_ = None
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        return fe.engineer_all(X, nbhd_stats=self.nbhd_stats_)


class RareEncoder(BaseEstimator, TransformerMixin):
    """Replace rare categories with "Other".

    A category is "rare" if its frequency in the training set is below
    `threshold` (default 1%). This prevents one-hot encoding from creating
    columns that appear in train but not test (or vice versa).

    Parameters
    ----------
    threshold : float, default 0.01
        Minimum frequency (as a proportion) for a category to be kept.
    fill_value : str, default "Other"
        Replacement label for rare categories.
    """

    def __init__(self, threshold=0.01, fill_value="Other"):
        self.threshold = threshold
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.frequent_cats_ = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.frequent_cats_[col] = set(freq[freq >= self.threshold].index)
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in X.columns:
            if col in self.frequent_cats_:
                mask = ~X[col].isin(self.frequent_cats_[col])
                X.loc[mask, col] = self.fill_value
        return X


class SkewnessTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to numeric columns with skewness above a threshold.

    Identifies skewed columns at fit() time (using training data only),
    then applies log1p consistently at transform() time.

    Parameters
    ----------
    threshold : float, default 0.75
        Absolute skewness above which log1p is applied.
    """

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        skew = X.apply(lambda col: col.dropna().skew())
        self.skewed_cols_ = skew[skew.abs() > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in self.skewed_cols_:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))
        return X
