

# src/make_dataset.py
import argparse
from typing import Optional
import pandas as pd
from config import (
    DATA_DIR,
    CALENDAR_FILE,
    PRICES_FILE,
    SALES_EVAL_FILE,
    SALES_VALID_FILE,
    PROCESSED_DIR,
    USE_PARQUET,
    ensure_dirs,
    N_SERIES_DEBUG,
    N_DAYS_DEBUG,
)

def _read_csv(path, usecols: Optional[list] = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols)

def load_raw(which: str = "evaluation") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: sales_wide, calendar, prices
    sales_wide columns include:
      id, item_id, dept_id, cat_id, store_id, state_id, d_1...d_N
    """
    sales_file = SALES_EVAL_FILE if which == "evaluation" else SALES_VALID_FILE

    sales_path = DATA_DIR / sales_file
    cal_path = DATA_DIR / CALENDAR_FILE
    price_path = DATA_DIR / PRICES_FILE

    print(f"[make_dataset] Loading sales: {sales_path.name}")
    sales = _read_csv(sales_path)

    if N_SERIES_DEBUG is not None:
        sales = sales.head(N_SERIES_DEBUG).copy()
        print(f"[make_dataset] DEBUG: limiting series to first {N_SERIES_DEBUG} rows")

    print(f"[make_dataset] Loading calendar: {cal_path.name}")
    calendar = _read_csv(cal_path)

    print(f"[make_dataset] Loading prices: {price_path.name}")
    prices = _read_csv(price_path)

    return sales, calendar, prices

def melt_sales(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """Wide -> long: one row per (id, d)."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    if N_DAYS_DEBUG is not None:
        d_cols = d_cols[:N_DAYS_DEBUG]
        print(f"[make_dataset] DEBUG: limiting to first {N_DAYS_DEBUG} day columns")

    long_df = sales_wide.melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="sales",
    )
    return long_df

def merge_calendar(long_df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar has columns like:
      date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, ...
    """
    # Keep all calendar cols except "date" as string conversion happens later
    merged = long_df.merge(calendar, on="d", how="left")
    return merged

def merge_prices(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Prices keyed by: store_id, item_id, wm_yr_wk
    Adds: sell_price
    """
    merged = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    return merged

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Sort for time series operations later
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    # Optional: fill missing sales with 0 (shouldn't happen)
    df["sales"] = df["sales"].fillna(0)

    return df

def save_processed(df: pd.DataFrame, which: str) -> str:
    ensure_dirs()
    out_base = PROCESSED_DIR / f"m5_long_{which}"

    if USE_PARQUET:
        out_path = str(out_base.with_suffix(".parquet"))
        df.to_parquet(out_path, index=False)
    else:
        out_path = str(out_base.with_suffix(".csv"))
        df.to_csv(out_path, index=False)

    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which",
        choices=["evaluation", "validation"],
        default="evaluation",
        help="Which sales matrix to process.",
    )
    args = parser.parse_args()

    sales_wide, calendar, prices = load_raw(args.which)

    print("[make_dataset] Melting sales wide -> long...")
    long_df = melt_sales(sales_wide)

    print("[make_dataset] Merging calendar...")
    df = merge_calendar(long_df, calendar)

    print("[make_dataset] Merging prices...")
    df = merge_prices(df, prices)

    print("[make_dataset] Finalizing...")
    df = finalize(df)

    print("[make_dataset] Saving processed dataset...")
    out_path = save_processed(df, args.which)

    print(f"[make_dataset] Done. Wrote: {out_path}")
    print(f"[make_dataset] Rows: {len(df):,} | Cols: {df.shape[1]}")

if __name__ == "__main__":
    main()


