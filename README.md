

# M5 Forecasting — LightGBM + HTML Forecast Dashboard + Inventory Simulation

This repo is a portfolio project built on the **M5 Forecasting** dataset (Walmart sales).  
It trains a **global LightGBM demand model** using calendar + price features, generates forecasts, and produces a set of **shareable HTML artifacts**:

- Interactive **Forecast Dashboard** (single HTML file)
- **Error Heatmaps** (where the model struggles by dept/store/state)
- **Inventory Simulation** (reorder point + safety stock + fill-rate outcomes)

> Note: Raw Kaggle datasets are not committed to GitHub. See “Data Setup” below.

---

## Project Highlights

### What’s modeled?
Daily unit sales per product-store series (`id`) with:
- calendar features (day of week, SNAP, events)
- price features (sell_price, price change)
- time series features (lags + rolling stats)

### What artifacts does it produce?
**Outputs (examples):**
- `outputs/dashboard/dashboard_v2_evaluation.html`  
  Interactive dropdown by `id` + forecast plot + KPI card (error + inventory policy)
- `outputs/reports/heatmap_dept_state_mae_evaluation.html`  
  Interactive MAE heatmap by `dept_id × state_id`
- `outputs/reports/inventory_sim_summary_evaluation.html`  
  Fill-rate distribution + worst-series table

---

## Repo Structure

m5-forecasting-accuracy/
data/ # raw Kaggle CSVs (ignored)
data_sample/ # optional tiny sample (committed)
outputs/
processed/ # parquet created by make_dataset.py
models/ # saved LightGBM models
forecasts/ # forecast csv outputs
dashboard/ # HTML dashboards
reports/ # heatmaps, summaries, inventory outputs
src/
config.py
make_dataset.py
features.py
train_lgbm.py
predict.py
dashboard_html.py
dashboard_html_v2.py
error_heatmaps.py
inventory_sim.py


