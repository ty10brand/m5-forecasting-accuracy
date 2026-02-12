

# src/inventory_sim.py
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from scipy.stats import norm

from config import (
    FORECASTS_DIR, REPORTS_DIR, ensure_dirs,
    TARGET_COL, DATE_COL, ID_COL
)

def simulate_one_series(dates, actual, forecast, lead_time=7, review_period=1,
                        service_level=0.95, holding_cost=1.0, stockout_cost=10.0):
    """
    Simple periodic-review base-stock style simulation using ROP policy.

    - lead_time: days until replenishment arrives
    - review_period: check reorder daily (1) for simplicity
    - safety stock computed from forecast error std over window
    Returns dict of metrics.
    """

    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    n = len(actual)

    # Estimate error sigma from forecast residuals
    residuals = actual - forecast
    sigma = np.std(residuals) + 1e-6

    # z for service level
    z = norm.ppf(service_level)

    # Demand during lead time (use forecast mean)
    # We use average forecast per day * lead_time
    mu_lt = np.mean(forecast) * lead_time

    # Safety stock (classic)
    ss = z * sigma * np.sqrt(lead_time)

    # Reorder point
    rop = mu_lt + ss

    # Simulation state
    on_hand = rop  # start with enough inventory
    pipeline = []  # list of (arrival_index, qty)

    stockouts = 0
    lost_units = 0.0
    total_demand = float(np.sum(actual))
    holding = 0.0
    stockout_penalty = 0.0

    for t in range(n):
        # Receive orders
        arrivals = [q for (arr_t, q) in pipeline if arr_t == t]
        if arrivals:
            on_hand += sum(arrivals)
        pipeline = [(arr_t, q) for (arr_t, q) in pipeline if arr_t != t]

        demand = actual[t]

        # Satisfy demand
        if on_hand >= demand:
            on_hand -= demand
        else:
            stockouts += 1
            lost = demand - on_hand
            lost_units += lost
            on_hand = 0.0

        # Costs (simple)
        holding += holding_cost * on_hand
        stockout_penalty += stockout_cost * (demand - min(demand, on_hand + demand))  # approx

        # Review + reorder daily
        if (t % review_period) == 0:
            inventory_position = on_hand + sum(q for (_, q) in pipeline)

            # Order enough to raise inventory_position to rop
            if inventory_position <= rop:
                order_qty = rop - inventory_position
                arrival_t = t + lead_time
                if arrival_t < n:
                    pipeline.append((arrival_t, order_qty))

    fill_rate = 1.0 if total_demand == 0 else (1.0 - lost_units / total_demand)

    return {
        "lead_time": lead_time,
        "service_level": service_level,
        "sigma": float(sigma),
        "safety_stock": float(ss),
        "rop": float(rop),
        "stockout_days": int(stockouts),
        "lost_units": float(lost_units),
        "total_demand": float(total_demand),
        "fill_rate": float(fill_rate),
        "avg_on_hand": float(holding / max(n, 1) / max(holding_cost, 1e-6)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["evaluation","validation"], default="evaluation")
    parser.add_argument("--lead_time", type=int, default=7)
    parser.add_argument("--service_level", type=float, default=0.95)
    parser.add_argument("--max_ids", type=int, default=300, help="Limit number of series simulated.")
    args = parser.parse_args()

    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    forecast_files = sorted(FORECASTS_DIR.glob(f"forecast_backtest_long_{args.which}_*.csv"))
    if not forecast_files:
        raise FileNotFoundError("No forecast_backtest_long file found. Run predict.py first.")
    pred_path = forecast_files[-1]
    print(f"[inventory_sim] Loading: {pred_path}")

    df = pd.read_csv(pred_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ID_COL, DATE_COL])

    ids = df[ID_COL].unique().tolist()[: args.max_ids]
    print(f"[inventory_sim] Simulating {len(ids)} series (max_ids={args.max_ids})")

    results = []
    policies = []

    for _id in ids:
        sub = df[df[ID_COL] == _id].sort_values(DATE_COL)
        dates = sub[DATE_COL].values
        actual = sub[TARGET_COL].values
        forecast = sub["yhat"].values

        res = simulate_one_series(
            dates, actual, forecast,
            lead_time=args.lead_time,
            service_level=args.service_level
        )
        res[ID_COL] = _id

        results.append({
            ID_COL: _id,
            "stockout_days": res["stockout_days"],
            "lost_units": res["lost_units"],
            "total_demand": res["total_demand"],
            "fill_rate": res["fill_rate"],
            "avg_on_hand": res["avg_on_hand"],
        })

        policies.append({
            ID_COL: _id,
            "lead_time": res["lead_time"],
            "service_level": res["service_level"],
            "sigma": res["sigma"],
            "safety_stock": res["safety_stock"],
            "rop": res["rop"],
        })

    results_df = pd.DataFrame(results)
    policies_df = pd.DataFrame(policies)

    out_results = REPORTS_DIR / f"inventory_sim_results_{args.which}.csv"
    out_policies = REPORTS_DIR / f"inventory_policy_{args.which}.csv"
    results_df.to_csv(out_results, index=False)
    policies_df.to_csv(out_policies, index=False)

    print(f"[inventory_sim] Wrote: {out_results}")
    print(f"[inventory_sim] Wrote: {out_policies}")

    # Simple HTML report: distribution of fill rate + table of worst series
    worst = results_df.sort_values("fill_rate").head(25)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=results_df["fill_rate"], nbinsx=30, name="Fill rate"))
    fig.update_layout(title=f"Fill Rate Distribution ({args.which})", xaxis_title="Fill rate", yaxis_title="count")

    table = go.Figure(
        data=[go.Table(
            header=dict(values=list(worst.columns)),
            cells=dict(values=[worst[c] for c in worst.columns])
        )]
    )
    table.update_layout(title="Worst 25 series by fill rate")

    # Combine two figs into one HTML (simple: write both sequentially)
    out_html = REPORTS_DIR / f"inventory_sim_summary_{args.which}.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<h1>Inventory Simulation Summary</h1>\n")
        f.write(f"<p>Lead time: {args.lead_time} days | Service level target: {args.service_level}</p>\n")
        f.write(pyo.plot(fig, include_plotlyjs=True, output_type="div"))
        f.write("<hr>\n")
        f.write(pyo.plot(table, include_plotlyjs=False, output_type="div"))

    print(f"[inventory_sim] Wrote: {out_html}")

if __name__ == "__main__":
    main()


