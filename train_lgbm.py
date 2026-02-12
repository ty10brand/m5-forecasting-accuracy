

# src/train_lgbm.py
import argparse
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

from config import (
    PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, ensure_dirs,
    TARGET_COL, DATE_COL, ID_COL,
    FORECAST_HORIZON, MIN_TRAIN_DAYS,
)
from features import prep_model_frame

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_true - y_pred) / denom)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["evaluation","validation"], default="evaluation")
    parser.add_argument("--cutoff", default=None, help="Cutoff date (YYYY-MM-DD). If None, uses max(date).")
    args = parser.parse_args()

    ensure_dirs()

    path = PROCESSED_DIR / f"m5_long_{args.which}.parquet"
    print(f"[train_lgbm] Loading processed: {path}")
    df = pd.read_parquet(path)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    max_date = df[DATE_COL].max()
    cutoff_date = pd.to_datetime(args.cutoff) if args.cutoff else max_date
    print(f"[train_lgbm] cutoff_date = {cutoff_date.date()} (max_date={max_date.date()})")

    # Define train/valid windows
    valid_start = cutoff_date - pd.Timedelta(days=FORECAST_HORIZON - 1)
    train_end = valid_start - pd.Timedelta(days=1)

    # Ensure we have enough history
    min_date = df[DATE_COL].min()
    if (train_end - min_date).days < MIN_TRAIN_DAYS:
        print(f"[train_lgbm] Warning: only {(train_end - min_date).days} train days. Consider increasing N_DAYS_DEBUG.")

    # Build features
    df_feat, feature_cols = prep_model_frame(df)

    # Drop rows where lags are missing (early history)
    # (Model needs complete lag features)
    needed = [c for c in feature_cols if c.startswith("lag_") or c.startswith("roll_")]
    before = len(df_feat)
    df_feat = df_feat.dropna(subset=needed)
    after = len(df_feat)
    print(f"[train_lgbm] Dropped {before-after:,} rows due to missing lag/roll features")

    train_mask = df_feat[DATE_COL] <= train_end
    valid_mask = (df_feat[DATE_COL] >= valid_start) & (df_feat[DATE_COL] <= cutoff_date)

    train_df = df_feat.loc[train_mask].copy()
    valid_df = df_feat.loc[valid_mask].copy()

    print(f"[train_lgbm] Train rows: {len(train_df):,} | Valid rows: {len(valid_df):,}")
    print(f"[train_lgbm] Feature count: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype("float32")

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET_COL].astype("float32")

    # LightGBM datasets
    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train, free_raw_data=False)

    params = {
        "objective": "poisson",           # demand is non-negative counts
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }

    print("[train_lgbm] Training...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train","valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50),
        ],
    )

    # Validation predictions + metrics
    preds = model.predict(X_valid, num_iteration=model.best_iteration)
    preds = np.clip(preds, 0, None)

    rmse = np.sqrt(np.mean((y_valid.values - preds) ** 2))
    s = smape(y_valid.values, preds)
    print(f"[train_lgbm] Valid RMSE: {rmse:.4f} | sMAPE: {s:.4f}")

    # Save model
    model_path = MODELS_DIR / f"lgbm_{args.which}_cutoff_{cutoff_date.date()}.pkl"
    joblib.dump({"model": model, "features": feature_cols, "cutoff": str(cutoff_date.date())}, model_path)
    print(f"[train_lgbm] Saved model: {model_path}")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"valid_metrics_{args.which}_cutoff_{cutoff_date.date()}.csv"
    pd.DataFrame([{"cutoff": str(cutoff_date.date()), "rmse": rmse, "smape": s, "n_valid": len(valid_df)}]).to_csv(report_path, index=False)
    print(f"[train_lgbm] Wrote report: {report_path}")

if __name__ == "__main__":
    main()


