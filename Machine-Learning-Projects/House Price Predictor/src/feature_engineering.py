"""Feature engineering functions for the Ames Housing dataset.

Each function takes a DataFrame and returns it with new columns added.
Functions are pure (no side effects on the input) -- they operate on copies.
"""
import numpy as np
import pandas as pd


def add_size_features(df):
    """Aggregate area and bathroom counts into composite signals."""
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBaths"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
        + df["WoodDeckSF"].fillna(0)
    )
    return df


def add_temporal_features(df):
    """Convert raw year columns to elapsed-time features."""
    df = df.copy()
    df["HouseAge"]          = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["YearsSinceRemodel"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["RecentRemodel"]     = (df["YearsSinceRemodel"] <= 10).astype(int)
    df["IsNewHouse"]        = (df["YearBuilt"] == df["YrSold"]).astype(int)
    return df


def add_interaction_features(df):
    """Capture multiplicative interactions between quality and size."""
    df = df.copy()
    df["QualxSF"]      = df["OverallQual"] * df["GrLivArea"]
    df["GarageScore"]  = df["GarageCars"].fillna(0) * df["GarageArea"].fillna(0)
    df["BasementScore"]= df["TotalBsmtSF"].fillna(0) * df["OverallQual"]
    return df


def add_neighborhood_stats(df, nbhd_stats=None):
    """Add neighborhood-level mean SalePrice encoding.

    Parameters
    ----------
    df         : DataFrame to transform
    nbhd_stats : pd.Series mapping Neighborhood to mean SalePrice.
                 Must be computed on TRAINING data only to avoid leakage.
    """
    df = df.copy()
    if nbhd_stats is not None:
        df["NeighborhoodMeanPrice"] = (
            df["Neighborhood"].map(nbhd_stats).fillna(nbhd_stats.mean())
        )
    else:
        df["NeighborhoodMeanPrice"] = 0
    return df


def engineer_all(df, nbhd_stats=None):
    """Apply all feature engineering steps in sequence."""
    df = add_size_features(df)
    df = add_temporal_features(df)
    df = add_interaction_features(df)
    df = add_neighborhood_stats(df, nbhd_stats=nbhd_stats)
    return df
