"""Regression metrics + inventory-oriented interpretation of forecast errors."""
import numpy as np


def rmsle(y_log, pred_log):
    """RMSLE = RMSE on the log1p scale (the competition metric)."""
    return float(np.sqrt(np.mean((np.asarray(pred_log) - np.asarray(y_log)) ** 2)))


def regression_report(y_log, pred_log) -> dict:
    y_log, pred_log = np.asarray(y_log), np.asarray(pred_log)
    y, p = np.expm1(y_log), np.clip(np.expm1(pred_log), 0, None)
    ss_res, ss_tot = np.sum((y_log - pred_log) ** 2), np.sum((y_log - y_log.mean()) ** 2)
    return {
        "RMSLE": rmsle(y_log, pred_log),
        "R2_log": float(1 - ss_res / ss_tot),
        "R2_units": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "MAE_units": float(np.mean(np.abs(y - p))),
        "RMSE_units": float(np.sqrt(np.mean((y - p) ** 2))),
    }


def inventory_report(y_log, pred_log) -> dict:
    """Convert forecast errors into product units.
    - Under-forecast (prediction below actual)  -> STOCKOUT risk
    - Over-forecast  (prediction above actual)  -> EXCESS stock / returns / waste risk
    """
    y, p = np.expm1(np.asarray(y_log)), np.clip(np.expm1(np.asarray(pred_log)), 0, None)
    under, over = np.clip(y - p, 0, None), np.clip(p - y, 0, None)
    total = y.sum()
    return {
        "actual_units": float(total),
        "under_forecast_units": float(under.sum()),
        "over_forecast_units": float(over.sum()),
        "under_forecast_pct_of_demand": float(100 * under.sum() / total),
        "over_forecast_pct_of_demand": float(100 * over.sum() / total),
    }
