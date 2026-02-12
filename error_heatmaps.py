

# src/error_heatmaps.py
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from config import (
    PROCESSED_DIR, FORECASTS_DIR, REPORTS_DIR, ensure_dirs,
    TARGET_COL, DATE_COL, ID_COL
)

def smape_vec(y, yhat):
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.abs(y - yhat) / denom

def make_heatmap(pivot_df: pd.DataFrame, title: str, out_path: str):
    z = pivot_df.values
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot_df.columns.astype(str),
            y=pivot_df.index.astype(str),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=pivot_df.columns.name or "",
        yaxis_title=pivot_df.index.name or "",
        height=700,
        margin=dict(l=120, r=40, t=80, b=60),
    )
    pyo.plot(fig, filename=out_path, auto_open=False, include_plotlyjs=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["evaluation","validation"], default="evaluation")
    args = parser.parse_args()

    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load forecasts (long)
    forecast_files = sorted(FORECASTS_DIR.glob(f"forecast_backtest_long_{args.which}_*.csv"))
    if not forecast_files:
        raise FileNotFoundError("No forecast_backtest_long file found. Run predict.py first.")
    pred_path = forecast_files[-1]

    print(f"[error_heatmaps] Loading predictions: {pred_path}")
    pred = pd.read_csv(pred_path)
    pred[DATE_COL] = pd.to_datetime(pred[DATE_COL])

    # Load processed to get hierarchy fields (dept/store/state/cat/item)
    proc_path = PROCESSED_DIR / f"m5_long_{args.which}.parquet"
    print(f"[error_heatmaps] Loading processed: {proc_path}")
    df = pd.read_parquet(proc_path, columns=[ID_COL, "item_id", "dept_id", "cat_id", "store_id", "state_id", DATE_COL])

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Merge hierarchy onto pred rows (match by id + date)
    merged = pred.merge(df, on=[ID_COL, DATE_COL], how="left")

    # Compute errors
    y = merged[TARGET_COL].astype(float).values
    yhat = merged["yhat"].astype(float).values

    merged["abs_error"] = np.abs(y - yhat)
    merged["sq_error"] = (y - yhat) ** 2
    merged["smape"] = smape_vec(y, yhat)

    # Aggregate summary
    def agg(group_cols, name):
        g = merged.groupby(group_cols).agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            smape=("smape", "mean"),
            mean_sales=(TARGET_COL, "mean"),
        ).reset_index()
        out_csv = REPORTS_DIR / f"error_summary_{name}_{args.which}.csv"
        g.to_csv(out_csv, index=False)
        print(f"[error_heatmaps] Wrote: {out_csv}")
        return g

    agg(["dept_id", "state_id"], "dept_state")
    agg(["store_id", "dept_id"], "store_dept")
    agg(["cat_id", "state_id"], "cat_state")

    # Heatmap 1: dept x state (MAE)
    dept_state = merged.groupby(["dept_id", "state_id"])["abs_error"].mean().reset_index()
    pivot1 = dept_state.pivot(index="dept_id", columns="state_id", values="abs_error").fillna(0)
    pivot1.index.name = "dept_id"
    pivot1.columns.name = "state_id"

    out1 = str(REPORTS_DIR / f"heatmap_dept_state_mae_{args.which}.html")
    make_heatmap(pivot1, f"MAE Heatmap — dept_id × state_id ({args.which})", out1)
    print(f"[error_heatmaps] Wrote: {out1}")

    # Heatmap 2: store x dept (MAE) — can get large; keep top 30 stores by volume
    store_volume = merged.groupby("store_id")[TARGET_COL].sum().sort_values(ascending=False)
    top_stores = store_volume.head(30).index.tolist()

    store_dept = merged[merged["store_id"].isin(top_stores)].groupby(["store_id", "dept_id"])["abs_error"].mean().reset_index()
    pivot2 = store_dept.pivot(index="store_id", columns="dept_id", values="abs_error").fillna(0)
    pivot2.index.name = "store_id"
    pivot2.columns.name = "dept_id"

    out2 = str(REPORTS_DIR / f"heatmap_store_dept_mae_top30stores_{args.which}.html")
    make_heatmap(pivot2, f"MAE Heatmap — store_id × dept_id (Top 30 stores by volume) ({args.which})", out2)
    print(f"[error_heatmaps] Wrote: {out2}")

if __name__ == "__main__":
    main()


