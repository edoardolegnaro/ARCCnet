import numpy as np
import pandas as pd


def split_by_noaa_group(df, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Split data by NOAA AR to prevent leakage.
    Ensures no NOAA AR appears in multiple splits.

    Returns
    -------
    train_mask, val_mask, test_mask : pd.Series
        Boolean masks for each split
    """
    np.random.seed(seed)

    unique_noaa = df["noaa_ar"].unique()
    n_noaa = len(unique_noaa)

    perm = np.random.permutation(unique_noaa)

    n_train = int(n_noaa * train_frac)
    n_val = int(n_noaa * val_frac)

    train_noaa = set(perm[:n_train])
    val_noaa = set(perm[n_train : n_train + n_val])
    test_noaa = set(perm[n_train + n_val :])

    train_mask = df["noaa_ar"].isin(train_noaa)
    val_mask = df["noaa_ar"].isin(val_noaa)
    test_mask = df["noaa_ar"].isin(test_noaa)

    return train_mask, val_mask, test_mask


def split_by_time(df, train_years=None, val_years=None, test_years=None):
    """
    Split data by year to simulate operational forecasting.

    Returns
    -------
    train_mask, val_mask, test_mask : pd.Series
        Boolean masks for each split
    """
    if train_years is None:
        train_years = [2011, 2017, 2018, 2019, 2020]
    if val_years is None:
        val_years = [2021]
    if test_years is None:
        test_years = [2022]

    df["year"] = pd.to_datetime(df["date"]).dt.year

    train_mask = df["year"].isin(train_years)
    val_mask = df["year"].isin(val_years)
    test_mask = df["year"].isin(test_years)

    return train_mask, val_mask, test_mask


def get_split(df, strategy="noaa", **kwargs):
    """
    Main entry point for splitting data.

    Parameters
    ----------
    df : pd.DataFrame
        Full manifest dataframe
    strategy : str
        'noaa' or 'time'
    **kwargs
        Additional arguments for split functions

    Returns
    -------
    train_mask, val_mask, test_mask : pd.Series
        Boolean masks for train/val/test splits
    """
    if strategy == "noaa":
        return split_by_noaa_group(df, **kwargs)
    elif strategy == "time":
        return split_by_time(df, **kwargs)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
