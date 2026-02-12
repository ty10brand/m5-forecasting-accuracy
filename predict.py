

# src/predict.py
import argparse
import joblib
import numpy as np
import pandas as pd

from config import (
    PROCESSED_DIR, MODELS_DIR, FORECASTS_DIR, ensure_dirs,
    TARGET_COL, DATE_COL, ID_COL, FORECAST_HORIZON
)
from features import prep_model_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["evaluation","validation"], default="evaluation")
    parser.add_argument("--model_path", default=None, help="Path to saved .pkl model. If None, uses latest pattern.")
    args = parser.parse_args()

    ensure_dirs()
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = PROCESSED_DIR / f"m5_long_{args.which}.parquet"
    print(f"[predict] Loading processed: {data_path}")
    df = pd.read_parquet(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Load model
    if args.model_path is None:
        # Simple default: use the one you just trained (cutoff = max date in df)
        cutoff = df[DATE_COL].max().date()
        model_path = MODELS_DIR / f"lgbm_{args.which}_cutoff_{cutoff}.pkl"
    else:
        model_path = args.model_path

    print(f"[predict] Loading model: {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    # Build feature frame
    df_feat, feat_cols_auto = prep_model_frame(df)

    # Ensure we use the exact training feature list
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")

    # We forecast the NEXT 28 days relative to the last date we have.
    last_date = df_feat[DATE_COL].max()
    print(f"[predict] Last observed date in dataset: {last_date.date()}")

    # For a simple MVP: use the last 28 days *rows* as “forecast horizon dates”
    # BUT we don't have future calendar/prices in your processed sample.
    # So we produce predictions for the last 28 days (a backtest-style forecast table)
    # This still powers the dashboard and error heatmaps right away.
    forecast_start = last_date - pd.Timedelta(days=FORECAST_HORIZON - 1)
    mask = (df_feat[DATE_COL] >= forecast_start) & (df_feat[DATE_COL] <= last_date)

    df_pred = df_feat.loc[mask].copy()

    # Drop rows with missing lags/rolls
    lag_roll_cols = [c for c in feature_cols if c.startswith("lag_") or c.startswith("roll_")]
    df_pred = df_pred.dropna(subset=lag_roll_cols)

    X = df_pred[feature_cols]
    yhat = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    yhat = np.clip(yhat, 0, None)

    df_pred["yhat"] = yhat

    # Pivot to F1..F28 format (per id)
    # map dates to F1..F28 based on relative ordering within horizon
    df_pred = df_pred.sort_values([ID_COL, DATE_COL])
    df_pred["h"] = df_pred.groupby(ID_COL).cumcount() + 1  # 1..28 (ideally)

    wide = df_pred.pivot_table(index=ID_COL, columns="h", values="yhat", aggfunc="first")
    # Rename to F1..F28
    wide.columns = [f"F{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    out_path = FORECASTS_DIR / f"forecast_backtest_{args.which}_{last_date.date()}.csv"
    wide.to_csv(out_path, index=False)
    print(f"[predict] Wrote: {out_path} | rows={len(wide):,} cols={wide.shape[1]}")

    # Also save a long version for dashboards
    out_long = FORECASTS_DIR / f"forecast_backtest_long_{args.which}_{last_date.date()}.csv"
    df_pred[[ID_COL, DATE_COL, TARGET_COL, "yhat"]].to_csv(out_long, index=False)
    print(f"[predict] Wrote: {out_long} | rows={len(df_pred):,}")

if __name__ == "__main__":
    main()


