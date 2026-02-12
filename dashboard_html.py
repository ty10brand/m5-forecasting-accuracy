

# src/dashboard_html.py
import argparse
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from config import (
    PROCESSED_DIR, FORECASTS_DIR, DASHBOARD_DIR, ensure_dirs,
    TARGET_COL, DATE_COL, ID_COL
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["evaluation","validation"], default="evaluation")
    parser.add_argument("--max_ids", type=int, default=200, help="Limit dropdown IDs for performance.")
    args = parser.parse_args()

    ensure_dirs()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    data_path = PROCESSED_DIR / f"m5_long_{args.which}.parquet"
    print(f"[dashboard] Loading processed: {data_path}")
    df = pd.read_parquet(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Find latest backtest forecast file (simple: pick the newest by name)
    # You can also hardcode the filename.
    forecast_files = sorted(FORECASTS_DIR.glob(f"forecast_backtest_long_{args.which}_*.csv"))
    if not forecast_files:
        raise FileNotFoundError("No forecast_backtest_long file found. Run predict.py first.")
    pred_path = forecast_files[-1]
    print(f"[dashboard] Loading predictions: {pred_path}")
    pred = pd.read_csv(pred_path)
    pred[DATE_COL] = pd.to_datetime(pred[DATE_COL])

    # Limit IDs for performance
    ids = sorted(pred[ID_COL].unique().tolist())
    ids = ids[: args.max_ids]
    print(f"[dashboard] Using {len(ids)} IDs in dropdown (max_ids={args.max_ids})")

    # Build traces: one history + one forecast per id, but only show first by default
    fig = go.Figure()

    default_id = ids[0]

    for i, _id in enumerate(ids):
        hist = df[df[ID_COL] == _id].sort_values(DATE_COL)
        fcst = pred[pred[ID_COL] == _id].sort_values(DATE_COL)

        # history trace
        fig.add_trace(
            go.Scatter(
                x=hist[DATE_COL],
                y=hist[TARGET_COL],
                mode="lines",
                name=f"{_id} actual",
                visible=(i == 0),
            )
        )
        # forecast trace
        fig.add_trace(
            go.Scatter(
                x=fcst[DATE_COL],
                y=fcst["yhat"],
                mode="lines",
                name=f"{_id} forecast",
                visible=(i == 0),
            )
        )

    # Dropdown controls visibility: each id corresponds to 2 traces
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
                    {"title": f"M5 Forecast Dashboard — {_id}"},
                ],
            )
        )

    fig.update_layout(
        title=f"M5 Forecast Dashboard — {default_id}",
        xaxis_title="Date",
        yaxis_title="Units",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.01,
                y=1.15,
                showactive=True,
            )
        ],
        height=650,
        margin=dict(l=40, r=40, t=90, b=40),
    )

    out_path = DASHBOARD_DIR / f"dashboard_{args.which}.html"
    pyo.plot(fig, filename=str(out_path), auto_open=False, include_plotlyjs=True)
    print(f"[dashboard] Wrote: {out_path}")

if __name__ == "__main__":
    main()


