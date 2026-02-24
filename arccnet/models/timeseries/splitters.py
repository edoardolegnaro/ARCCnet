import numpy as np
import pandas as pd


def _build_split_dict(df, train_mask, val_mask, test_mask, strategy, metadata=None):
    """Build a standard split dictionary shared by train/eval."""
    train_mask = pd.Series(train_mask, index=df.index).astype(bool)
    val_mask = pd.Series(val_mask, index=df.index).astype(bool)
    test_mask = pd.Series(test_mask, index=df.index).astype(bool)

    out = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "strategy": strategy,
        "train_df": df[train_mask].copy(),
        "val_df": df[val_mask].copy(),
        "test_df": df[test_mask].copy(),
    }
    # Backward-compatible aliases.
    out["train"] = out["train_df"]
    out["val"] = out["val_df"]
    out["test"] = out["test_df"]
    if metadata:
        out.update(metadata)
    return out


def split_by_noaa_group(df, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Split data by NOAA AR to reduce active-region identity leakage.
    Ensures each NOAA AR appears in exactly one split.
    """
    if "noaa_ar" not in df.columns:
        raise ValueError("Expected 'noaa_ar' column in manifest for NOAA split")

    rng = np.random.default_rng(seed)
    unique_noaa = np.array(sorted(df["noaa_ar"].unique()))
    n_noaa = len(unique_noaa)
    if n_noaa == 0:
        raise ValueError("No NOAA AR values found in manifest")

    perm = rng.permutation(unique_noaa)
    n_train = int(n_noaa * train_frac)
    n_val = int(n_noaa * val_frac)

    # Keep at least one group for test when feasible.
    if n_noaa >= 3:
        n_train = min(max(n_train, 1), n_noaa - 2)
        n_val = min(max(n_val, 1), n_noaa - n_train - 1)

    train_noaa = set(perm[:n_train].tolist())
    val_noaa = set(perm[n_train : n_train + n_val].tolist())
    test_noaa = set(perm[n_train + n_val :].tolist())

    train_mask = df["noaa_ar"].isin(train_noaa)
    val_mask = df["noaa_ar"].isin(val_noaa)
    test_mask = df["noaa_ar"].isin(test_noaa)

    metadata = {
        "train_noaa": sorted(train_noaa),
        "val_noaa": sorted(val_noaa),
        "test_noaa": sorted(test_noaa),
    }
    return _build_split_dict(df, train_mask, val_mask, test_mask, strategy="noaa", metadata=metadata)


def split_by_time(df, train_years=None, val_years=None, test_years=None):
    """
    Split data by year to simulate operational forecasting.
    """
    if "date" not in df.columns:
        raise ValueError("Expected 'date' column in manifest for time split")

    if train_years is None:
        train_years = [2011, 2017, 2018, 2019, 2020]
    if val_years is None:
        val_years = [2021]
    if test_years is None:
        test_years = [2022]

    years = pd.to_datetime(df["date"], errors="coerce").dt.year
    train_mask = years.isin(train_years)
    val_mask = years.isin(val_years)
    test_mask = years.isin(test_years)

    metadata = {
        "train_years": list(train_years),
        "val_years": list(val_years),
        "test_years": list(test_years),
    }
    return _build_split_dict(df, train_mask, val_mask, test_mask, strategy="time", metadata=metadata)


def get_split(df, strategy="noaa", **kwargs):
    """
    Main entry point for split generation.

    Returns
    -------
    dict
        Keys: train_mask, val_mask, test_mask, train_df, val_df, test_df, strategy
    """
    if strategy == "noaa":
        return split_by_noaa_group(df, **kwargs)
    if strategy == "time":
        return split_by_time(df, **kwargs)
    raise ValueError(f"Unknown split strategy: {strategy}")
