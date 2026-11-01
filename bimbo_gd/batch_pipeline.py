"""Train Gradient Descent BATCH BY BATCH (one week, ~1M rows each) reading straight from SQL.

"Expanding window" idea - only ONE batch is ever in RAM:
    first batch (earliest week)  -> only builds historical statistics (no training)
    middle batches               -> features = statistics of PREVIOUS weeks; measure R² (before learning), then learn
    last batch (latest week)     -> held out for validation (the model has never seen it)
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config
from .data_loader import _DOWNCAST
from .features import FEATURES, HistoricalFeatureBuilder, log_target
from .gradient_descent import LinearRegressionGD
from .metrics import inventory_report, regression_report


# ---------------------------------------------------------------- SQL reading
def _where(sample_mod=1, week=None) -> str:
    conds = []
    if week is not None:
        conds.append(f"Week = {int(week)}")
    if sample_mod and sample_mod > 1:               # customer-based sampling keeps each customer's full history
        conds.append(f"Customer_ID % {int(sample_mod)} = 0")
    return ("WHERE " + " AND ".join(conds)) if conds else ""


def count_by_week(engine, table, sample_mod=1) -> pd.DataFrame:
    sql = f"SELECT Week, COUNT(*) AS n_rows FROM {table} {_where(sample_mod)} GROUP BY Week ORDER BY Week"
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def _cast(chunk: pd.DataFrame) -> pd.DataFrame:
    return chunk.astype({c: t for c, t in _DOWNCAST.items() if c in chunk.columns})


def fetch_week(engine, table, columns, week, chunksize=250_000, sample_mod=1) -> pd.DataFrame:
    """Read one batch (one week) from SQL in small chunks, then concatenate."""
    sql = f"SELECT {', '.join(columns)} FROM {table} {_where(sample_mod, week)}"
    parts = []
    with engine.connect().execution_options(stream_results=True) as conn:
        for ch in pd.read_sql(text(sql), conn, chunksize=chunksize):
            parts.append(_cast(ch))
    return pd.concat(parts, ignore_index=True)


def iter_chunks(engine, table, columns, chunksize=500_000, sample_mod=1):
    """Iterate over a whole table chunk by chunk (used for the test table; nothing is kept in RAM)."""
    sql = f"SELECT {', '.join(columns)} FROM {table} {_where(sample_mod)}"
    with engine.connect().execution_options(stream_results=True) as conn:
        for ch in pd.read_sql(text(sql), conn, chunksize=chunksize):
            yield _cast(ch)


def table_columns(engine, table) -> list[str]:
    with engine.connect() as conn:
        return list(pd.read_sql(text(f"SELECT * FROM {table} WHERE 1=0"), conn).columns)


# ---------------------------------------------------------------- per-product table
def product_table(val: pd.DataFrame) -> pd.DataFrame:
    """One row per Product_ID: actual vs predicted demand (validation week)."""
    g = val.groupby("Product_ID").agg(
        n_rows=("Product_ID", "size"),
        Actual_Total=("Adjusted_Demand_Units", "sum"), Predicted_Total=("Predicted_Demand", "sum"),
        Actual_Mean=("Adjusted_Demand_Units", "mean"), Predicted_Mean=("Predicted_Demand", "mean"),
    ).reset_index()
    g["Error_%"] = 100 * (g["Predicted_Total"] - g["Actual_Total"]) / g["Actual_Total"].replace(0, np.nan)
    return g.sort_values("Actual_Total", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- training
def train_in_batches(engine, table=None, lr=0.05, epochs_per_batch=5, batch_size=4096,
                     l2=0.0, sample_mod=None, log=print) -> dict:
    table = table or config.TRAIN_TABLE
    sample_mod = config.BATCH_SAMPLE_MOD if sample_mod is None else sample_mod
    cnt = count_by_week(engine, table, sample_mod)
    weeks = cnt["Week"].astype(int).tolist()
    log(f"Sampling: Customer_ID % {sample_mod} = 0 -> {cnt.n_rows.sum():,} rows, "
        f"largest batch {cnt.n_rows.max():,} rows")
    if len(weeks) < 3:
        raise ValueError(f"Need >= 3 weeks (seed / train / validation), found only {weeks}")
    seed_w, train_ws, val_w = weeks[0], weeks[1:-1], weeks[-1]
    log(f"Batches (weeks): {weeks}\n  seed={seed_w} | train={train_ws} | validation={val_w}")
    hyper = {"learning_rate": lr, "epochs_per_batch": epochs_per_batch, "mini_batch_size": batch_size,
             "l2": l2, "sample_mod": sample_mod, "method": "mini_batch"}
    log(f"Hyper-parameters: {hyper}")

    builder, model, rows = HistoricalFeatureBuilder(), None, []

    t0 = time.time()
    df = fetch_week(engine, table, config.TRAIN_COLUMNS, seed_w, sample_mod=sample_mod)
    builder.update(df)
    log(f"[Batch week {seed_w}] {len(df):,} rows -> only builds historical statistics ({time.time()-t0:.0f}s)")
    del df

    for w in train_ws:
        t0 = time.time()
        df = fetch_week(engine, table, config.TRAIN_COLUMNS, w, sample_mod=sample_mod)
        X, y = builder.transform(df).to_numpy(), log_target(df[config.TARGET])
        if model is None:
            model = LinearRegressionGD(method="mini_batch", learning_rate=lr,
                                       batch_size=batch_size, l2=l2, verbose=False).init_scaler(X, y)
        rep = regression_report(y, model.predict(X))      # measured BEFORE learning = true out-of-sample
        loss = model.partial_fit(X, y, n_epochs=epochs_per_batch)
        builder.update(df)                                 # only after learning is this week added to history
        rows.append({"week": w, "rows": len(df), "epochs": epochs_per_batch, "learning_rate": lr,
                     "R2_before_learning": rep["R2_log"], "RMSLE_before_learning": rep["RMSLE"],
                     "train_loss_after": loss, "seconds": round(time.time() - t0, 1)})
        log(f"[Batch week {w}] {len(df):,} rows | R² before learning = {rep['R2_log']:.4f} "
            f"| loss after learning = {loss:.5f} | {time.time()-t0:.0f}s")
        del df, X, y

    # ---- validation on the last batch (never trained on)
    df = fetch_week(engine, table, config.TRAIN_COLUMNS, val_w, sample_mod=sample_mod)
    Xdf = builder.transform(df)
    X, y = Xdf.to_numpy(), log_target(df[config.TARGET])
    pred = model.predict(X)
    metrics = regression_report(y, pred)
    inv = inventory_report(y, pred)
    base = regression_report(y, np.full_like(y, model.b))
    log(f"[Validation week {val_w}] {len(df):,} rows | R² = {metrics['R2_log']:.4f} | RMSLE = {metrics['RMSLE']:.4f}")

    val = pd.DataFrame({
        "Product_ID": df["Product_ID"].to_numpy(), "Customer_ID": df["Customer_ID"].to_numpy(),
        "Adjusted_Demand_Units": df[config.TARGET].to_numpy(),
        "Predicted_Demand": np.clip(np.expm1(pred), 0, None),
        "y_log": y, "pred_log": pred})
    val["Error"] = val["Predicted_Demand"] - val["Adjusted_Demand_Units"]
    builder.update(df)          # add the validation week to the history (used when scoring the test set)
    del df
    return {"model": model, "builder": builder, "log": pd.DataFrame(rows), "val": val, "X_val": Xdf,
            "metrics": metrics, "baseline_metrics": base, "inventory": inv, "sample_mod": sample_mod,
            "hyperparameters": hyper, "weeks": {"seed": seed_w, "train": train_ws, "val": val_w}}


# ---------------------------------------------------------------- save results
def save_artifacts(res: dict):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    res["model"].save(config.BATCH_MODEL_PATH, FEATURES)
    with open(config.BATCH_BUILDER_PATH, "wb") as f:
        pickle.dump(res["builder"], f, protocol=pickle.HIGHEST_PROTOCOL)
    res["log"].to_csv(config.BATCH_LOG_PATH, index=False)
    config.BATCH_METRICS_PATH.write_text(json.dumps(
        {"weeks": res["weeks"], "sample_mod": res["sample_mod"], "hyperparameters": res["hyperparameters"],
         "validation": res["metrics"], "baseline": res["baseline_metrics"],
         "inventory": res["inventory"]}, indent=2))
    res["X_val"].agg(["mean", "std"]).T.to_csv(config.VAL_FEATURE_STATS_PATH)
    bins = np.linspace(0, 8, 81)
    v = res["val"]
    pd.DataFrame({"bin_left": bins[:-1],
                  "actual": np.histogram(v["y_log"], bins, density=True)[0],
                  "predicted": np.histogram(v["pred_log"], bins, density=True)[0]}
                 ).to_csv(config.VAL_DIST_PATH, index=False)
    product_table(v).to_csv(config.VAL_PRODUCT_PATH, index=False)   # open in Excel
    print("Saved model, historical statistics, logs, metrics and per-product table to outputs/")


def load_artifacts():
    model, feats = LinearRegressionGD.load(config.BATCH_MODEL_PATH)
    with open(config.BATCH_BUILDER_PATH, "rb") as f:
        builder = pickle.load(f)
    return model, builder, feats
