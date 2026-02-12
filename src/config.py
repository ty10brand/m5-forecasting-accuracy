

# src/config.py
from pathlib import Path

# ---- Paths (edit ROOT if needed) ----
# ROOT should point to your project folder: .../m5-forecasting-accuracy
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATA_SAMPLE_DIR = ROOT / "data_sample"
OUTPUTS_DIR = ROOT / "outputs"

PROCESSED_DIR = OUTPUTS_DIR / "processed"
MODELS_DIR = OUTPUTS_DIR / "models"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DASHBOARD_DIR = OUTPUTS_DIR / "dashboard"

# ---- Raw file names (no extensions shown in Explorer sometimes) ----
# If your files are named "calendar.csv" etc, set these to include ".csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"
SALES_EVAL_FILE = "sales_train_evaluation.csv"
SALES_VALID_FILE = "sales_train_validation.csv"
SUBMISSION_FILE = "sample_submission.csv"

# ---- Processing options ----
# Use Parquet if you install pyarrow (recommended). Otherwise fallback to CSV.
USE_PARQUET = True

# Optional: reduce memory while prototyping
# If not None, only load this many series (rows) from the sales matrix
N_SERIES_DEBUG = 2000  # e.g., 500 for quick tests

# Optional: limit number of days loaded (from the start) for quick tests
N_DAYS_DEBUG = 730  # e.g., 365

def ensure_dirs():
    """Create output directories if they don't exist."""
    for p in [OUTPUTS_DIR, PROCESSED_DIR, MODELS_DIR, FORECASTS_DIR, REPORTS_DIR, DASHBOARD_DIR]:
        p.mkdir(parents=True, exist_ok=True)


# ---- Modeling setup ----
TARGET_COL = "sales"
DATE_COL = "date"
ID_COL = "id"

# Backtest settings (works for your 730-day sample)
FORECAST_HORIZON = 28
MIN_TRAIN_DAYS = 365  # keep >= 365 so lags/rolls have history

# Feature engineering
LAGS = [1, 7, 28]
ROLL_WINDOWS = [7, 28]




