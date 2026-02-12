

# src/features.py
import numpy as np
import pandas as pd
from config import (
    TARGET_COL, DATE_COL, ID_COL,
    LAGS, ROLL_WINDOWS,
)

CAL_COLS_GUESS = [
    "wday", "month", "year", "wm_yr_wk",
    "snap_CA", "snap_TX", "snap_WI",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df["dow"] = df[DATE_COL].dt.dayofweek.astype("int16")  # 0=Mon
    df["weekofyear"] = df[DATE_COL].dt.isocalendar().week.astype("int16")
    df["day"] = df[DATE_COL].dt.day.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sell_price" not in df.columns:
        return df

    # price change vs previous week (by id, ordered by date)
    df["price_lag_7"] = df.groupby(ID_COL)["sell_price"].shift(7)
    df["price_change_7"] = (df["sell_price"] - df["price_lag_7"]).astype("float32")

    # rolling price mean (to proxy “normal price”)
    df["price_roll_28"] = (
        df.groupby(ID_COL)["sell_price"].shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    ).astype("float32")

    df["price_rel_28"] = (df["sell_price"] / (df["price_roll_28"] + 1e-6)).astype("float32")
    return df

def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      lag_{k}
      roll_mean_{w}
      roll_std_{w}
    All computed per-series (id) with shift(1) so no leakage.
    """
    df = df.copy()
    df = df.sort_values([ID_COL, DATE_COL])

    # Lags
    for k in LAGS:
        df[f"lag_{k}"] = df.groupby(ID_COL)[TARGET_COL].shift(k)

    # Rolling stats based on shifted target
    shifted = df.groupby(ID_COL)[TARGET_COL].shift(1)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = (
            shifted.groupby(df[ID_COL]).rolling(w).mean().reset_index(level=0, drop=True)
        )
        df[f"roll_std_{w}"] = (
            shifted.groupby(df[ID_COL]).rolling(w).std().reset_index(level=0, drop=True)
        )

    # Fill std NaN with 0 (std undefined early)
    for w in ROLL_WINDOWS:
        df[f"roll_std_{w}"] = df[f"roll_std_{w}"].fillna(0)

    return df

def prep_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
      df_feat (with features + target)
      feature_cols list
    """
    df = df.copy()

    # Basic types: category for IDs and event strings
    for c in ["item_id","dept_id","cat_id","store_id","state_id"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    for c in ["event_name_1","event_type_1","event_name_2","event_type_2"]:
        if c in df.columns:
            df[c] = df[c].fillna("None").astype("category")

    df = add_time_features(df)
    df = add_price_features(df)
    df = add_lag_roll_features(df)

    # Candidate features
    feature_cols = []

    # IDs (static)
    for c in ["item_id","dept_id","cat_id","store_id","state_id"]:
        if c in df.columns:
            feature_cols.append(c)

    # calendar / snap / events (if present)
    for c in CAL_COLS_GUESS:
        if c in df.columns and c != DATE_COL:
            feature_cols.append(c)

    # engineered time
    feature_cols += ["dow", "weekofyear", "day", "is_weekend"]

    # engineered price
    for c in ["sell_price", "price_change_7", "price_rel_28"]:
        if c in df.columns:
            feature_cols.append(c)

    # lags/rolls
    for k in LAGS:
        feature_cols.append(f"lag_{k}")
    for w in ROLL_WINDOWS:
        feature_cols.append(f"roll_mean_{w}")
        feature_cols.append(f"roll_std_{w}")

    # Drop features that may not exist (safety)
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols


