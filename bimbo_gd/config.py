"""Central configuration - values come from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DB_DIALECT = os.getenv("DB_DIALECT", "mssql").lower()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT") or None
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER") or None
DB_PASSWORD = os.getenv("DB_PASSWORD") or None
DB_TRUSTED = os.getenv("DB_TRUSTED_CONNECTION", "false").lower() == "true"
DB_ODBC_DRIVER = os.getenv("DB_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
DB_SQLITE_PATH = ROOT / os.getenv("DB_SQLITE_PATH", "data/bimbo.db")

TRAIN_TABLE = os.getenv("TRAIN_TABLE", "train_clean")
TEST_TABLE = os.getenv("TEST_TABLE", "test_clean")
SAMPLE_MOD = int(os.getenv("SAMPLE_MOD", "20"))

# Batch pipeline: keep 1/BATCH_SAMPLE_MOD of the customers (full history of each kept customer).
# 10 -> ~7.4M rows -> 7 weekly batches of ~1M rows.
BATCH_SAMPLE_MOD = int(os.getenv("BATCH_SAMPLE_MOD", "10"))

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
TRAIN_SAMPLE_PATH = PROCESSED_DIR / "train_sample.parquet"
TEST_SAMPLE_PATH = PROCESSED_DIR / "test_sample.parquet"

TRAIN_COLUMNS = [
    "Week", "Agency_ID", "Channel_ID", "Route_SAK", "Customer_ID", "Product_ID",
    "Sales_Units_Today", "Sales_Today", "Returns_Units_Next", "Returns_Next",
    "Adjusted_Demand_Units",
]
TEST_COLUMNS = ["id", "Week", "Agency_ID", "Channel_ID", "Route_SAK", "Customer_ID", "Product_ID"]
TARGET = "Adjusted_Demand_Units"

# Time-based split for the small-sample pipeline (no random shuffling: this is a forecasting problem)
HISTORY_WEEKS = [3, 4, 5, 6]   # weeks used to compute historical statistics (features)
TRAIN_WEEKS = [7, 8]           # weeks used to train Gradient Descent
VAL_WEEKS = [9]                # validation week

# ---- Batch pipeline outputs (one weekly batch at a time) ----
BATCH_MODEL_PATH = OUTPUT_DIR / "model_batches.npz"
BATCH_BUILDER_PATH = OUTPUT_DIR / "history_stats.pkl"
BATCH_LOG_PATH = OUTPUT_DIR / "batch_training_log.csv"
BATCH_METRICS_PATH = OUTPUT_DIR / "batch_validation_metrics.json"
VAL_FEATURE_STATS_PATH = OUTPUT_DIR / "val_feature_stats.csv"
VAL_DIST_PATH = OUTPUT_DIR / "val_distribution.csv"
VAL_PRODUCT_PATH = OUTPUT_DIR / "product_predictions_validation.csv"
TEST_PRED_PATH = OUTPUT_DIR / "test_predictions.csv"
TEST_PRODUCT_PATH = OUTPUT_DIR / "test_predictions_by_product.csv"

# =====================================================================
# FINAL MODEL: tables already filtered in SQL (train_filtered / test_filtered), ALL rows used, no sampling
# =====================================================================
FILTERED_TRAIN_TABLE = os.getenv("FILTERED_TRAIN_TABLE", "dbo.train_filtered")
FILTERED_TEST_TABLE = os.getenv("FILTERED_TEST_TABLE", "dbo.test_filtered")

SEED_WEEK = 3                  # week 3 only supplies history (features of week 4 need an earlier week)
TRAIN_WEEKS_FINAL = [4, 5, 6, 7, 8]
VAL_WEEK_FINAL = 9
TEST_WEEKS_FINAL = [10, 11]
SUB_BATCH_ROWS = 1_000_000     # no week is ever processed in one piece: chunks of at most this many rows

FINAL_DIR = OUTPUT_DIR / "final"
FINAL_CACHE_DIR = PROCESSED_DIR / "final"
FINAL_MODEL_PATH = FINAL_DIR / "model_final.npz"
FINAL_BUILDER_PATH = FINAL_DIR / "history_stats_final.pkl"
FINAL_METRICS_PATH = FINAL_DIR / "metrics_final.json"
FINAL_PRODUCT_PATH = FINAL_DIR / "product_results_week9.csv"
FINAL_COEF_PATH = FINAL_DIR / "coefficients_final.csv"
FINAL_WEEKLY_PATH = FINAL_DIR / "weekly_product_summary.csv"
FINAL_REF_PATH = FINAL_DIR / "reference_check.npz"
FINAL_TRAINLOG_PATH = FINAL_DIR / "training_log.csv"
FINAL_DIST_PATH = FINAL_DIR / "val_distribution.csv"
FINAL_EDA_PATH = FINAL_DIR / "eda_summary.json"
FINAL_TEST_ROWS_PATH = FINAL_DIR / "test_predictions_rows.csv"
FINAL_TEST_PRODUCT_PATH = FINAL_DIR / "test_predictions_by_product_week.csv"
FINAL_TEST_SUMMARY_PATH = FINAL_DIR / "test_summary.json"
