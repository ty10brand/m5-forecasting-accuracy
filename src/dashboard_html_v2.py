

# src/dashboard_html_v2.py
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from config import (
    PROCESSED_DIR, FORECASTS_DIR, REPORTS_DIR, DASHBOARD_DIR, ensure_dirs,
    TARGET_COL, DATE_COL, ID_COL
)

def smape(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y - yhat) / denom))

def kpi_block(metrics: dict) -> str:
    # Render a compact multi-line KPI "card" as HTML-ish text inside annotation
    # (Plotly supports <br>)
    def fmt(x, nd=3):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{nd}f}"
        return str(x)

    lines = [
        "<b>Model Error (Backtest 28d)</b>",
        f"MAE: {fmt(metrics.get('mae'), 3)}",
        f"sMAPE: {fmt(metrics.get('smape'), 3)}",
        f"RMSE: {fmt(metrics.get('rmse'), 3)}",
        "",
        "<b>Inventory (Sim)</b>",
        f"Fill rate: {fmt(metrics.get('fill_rate'), 3)}",
        f"Stockout days: {fmt(metrics.get('stockout_days'))}",
        f"Lost units: {fmt(metrics.get('lost_units'), 2)}",
        "",
        "<b>Policy</b>",
        f"ROP: {fmt(metrics.get('rop'), 2)}",
        f"Safety stock: {fmt(metrics.get('safety_stock'), 2)}",
        f"Sigma(err): {fmt(metrics.get('sigma'), 3)}",
    ]
    return "<br>".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["evaluation","validation"], default="evaluation")
    parser.add_argument("--max_ids", type=int, default=200, help="Limit dropdown IDs for performance.")
    args = parser.parse_args()

    ensure_dirs()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load processed history (for plotting) ----
    proc_path = PROCESSED_DIR / f"m5_long_{args.which}.parquet"
    print(f"[dashboard_v2] Loading processed: {proc_path}")
    hist = pd.read_parquet(proc_path, columns=[ID_COL, DATE_COL, TARGET_COL])
    hist[DATE_COL] = pd.to_datetime(hist[DATE_COL])
    hist = hist.sort_values([ID_COL, DATE_COL])

    # ---- Load backtest predictions (actual + yhat) ----
    forecast_files = sorted(FORECASTS_DIR.glob(f"forecast_backtest_long_{args.which}_*.csv"))
    if not forecast_files:
        raise FileNotFoundError("No forecast_backtest_long file found. Run predict.py first.")
    pred_path = forecast_files[-1]
    print(f"[dashboard_v2] Loading predictions: {pred_path}")
    pred = pd.read_csv(pred_path)
    pred[DATE_COL] = pd.to_datetime(pred[DATE_COL])
    pred = pred.sort_values([ID_COL, DATE_COL])

    # ---- Load inventory artifacts (optional but expected) ----
    policy_path = REPORTS_DIR / f"inventory_policy_{args.which}.csv"
    sim_path = REPORTS_DIR / f"inventory_sim_results_{args.which}.csv"

    print(f"[dashboard_v2] Loading inventory policy: {policy_path}")
    pol = pd.read_csv(policy_path)

    print(f"[dashboard_v2] Loading inventory sim results: {sim_path}")
    sim = pd.read_csv(sim_path)

    # ---- Choose IDs for dropdown based on pred file (has yhat) ----
    ids = sorted(pred[ID_COL].unique().tolist())
    ids = ids[: args.max_ids]
    print(f"[dashboard_v2] Using {len(ids)} IDs in dropdown (max_ids={args.max_ids})")

    # ---- Precompute per-id metrics ----
    metrics_by_id = {}
    pred_group = pred.groupby(ID_COL)
    pol_map = pol.set_index(ID_COL).to_dict(orient="index")
    sim_map = sim.set_index(ID_COL).to_dict(orient="index")

    for _id in ids:
        g = pred_group.get_group(_id)
        y = g[TARGET_COL].astype(float).values
        yhat = g["yhat"].astype(float).values

        mae = float(np.mean(np.abs(y - yhat)))
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        sm = smape(y, yhat)

        m = {
            "mae": mae,
            "rmse": rmse,
            "smape": sm,
        }

        # inventory sim results
        if _id in sim_map:
            m.update(sim_map[_id])

        # policy
        if _id in pol_map:
            m.update(pol_map[_id])

        metrics_by_id[_id] = m

    # ---- Build Plotly figure: two traces per id (history + forecast) ----
    fig = go.Figure()
    default_id = ids[0]

    for i, _id in enumerate(ids):
        h = hist[hist[ID_COL] == _id].sort_values(DATE_COL)
        f = pred[pred[ID_COL] == _id].sort_values(DATE_COL)

        fig.add_trace(
            go.Scatter(
                x=h[DATE_COL],
                y=h[TARGET_COL],
                mode="lines",
                name=f"{_id} actual",
                visible=(i == 0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=f[DATE_COL],
                y=f["yhat"],
                mode="lines",
                name=f"{_id} forecast (backtest)",
                visible=(i == 0),
            )
        )

    # ---- Annotation “KPI card” (updated per dropdown selection) ----
    def make_annotations(_id):
        block = kpi_block(metrics_by_id.get(_id, {}))
        return [
            dict(
                x=0.01, y=1.15, xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                align="left",
                showarrow=False,
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                text=block,
                font=dict(size=12),
            )
        ]

    # ---- Dropdown buttons: toggle visibility + update title + KPI annotation ----
    buttons = []
    for i, _id in enumerate(ids):
        vis = [False] * (2 * len(ids))
        vis[2*i] = True
        vis[2*i + 1] = True

        buttons.append(
            dict(
                label=_id,
                method="update",
                args=[
                    {"visible": vis},
                    {"title": f"M5 Dashboard v2 — {_id}", "annotations": make_annotations(_id)},
                ],
            )
        )

    fig.update_layout(
        title=f"M5 Dashboard v2 — {default_id}",
        xaxis_title="Date",
        yaxis_title="Units",
        height=720,
        margin=dict(l=50, r=40, t=110, b=50),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.01,
                y=1.27,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ],
        annotations=make_annotations(default_id),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out_path = DASHBOARD_DIR / f"dashboard_v2_{args.which}.html"
    pyo.plot(fig, filename=str(out_path), auto_open=False, include_plotlyjs=True)
    print(f"[dashboard_v2] Wrote: {out_path}")

if __name__ == "__main__":
    main()


