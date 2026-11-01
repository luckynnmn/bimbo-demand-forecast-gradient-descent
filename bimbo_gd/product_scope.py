"""Helpers for the 20-product scope: filtered SQL reads (with local parquet cache), exact least-squares
accumulation (to cross-check Gradient Descent), formula printing and per-product reports."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config
from .batch_pipeline import _cast


def _in_clause(products) -> str:
    return "Product_ID IN (" + ", ".join(str(int(p)) for p in products) + ")"


def _where(products=None, week=None, sample_mod=1) -> str:
    conds = []
    if products:
        conds.append(_in_clause(products))
    if week is not None:
        conds.append(f"Week = {int(week)}")
    if sample_mod and sample_mod > 1:
        conds.append(f"Customer_ID % {int(sample_mod)} = 0")
    return ("WHERE " + " AND ".join(conds)) if conds else ""


def detect_products(engine, table) -> list:
    """Distinct Product_IDs present in a (filtered) table."""
    with engine.connect() as conn:
        return [int(x) for x in pd.read_sql(text(f"SELECT DISTINCT Product_ID FROM {table} ORDER BY Product_ID"), conn)["Product_ID"]]


def count_rows(engine, table, products=None, sample_mod=1) -> pd.DataFrame:
    """Rows per (Week, Product_ID) - one scan of the table (products=None means every row)."""
    sql = (f"SELECT Week, Product_ID, COUNT(*) AS n_rows FROM {table} {_where(products, None, sample_mod)} "
           "GROUP BY Week, Product_ID ORDER BY Week, Product_ID")
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def fetch_week_products(engine, table, columns, week, products=None, sample_mod=1, cache_dir=None,
                        tag="train", chunksize=250_000) -> pd.DataFrame:
    """Read one week from SQL in chunks (all rows, or only the given products). The result is cached as parquet,
    so re-running a notebook does not query the database again."""
    cache = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache = Path(cache_dir) / f"{tag}_week{int(week)}_mod{int(sample_mod)}_n{len(products) if products else "all"}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
    sql = f"SELECT {', '.join(columns)} FROM {table} {_where(products, week, sample_mod)}"
    parts = []
    with engine.connect().execution_options(stream_results=True) as conn:
        for ch in pd.read_sql(text(sql), conn, chunksize=chunksize):
            parts.append(_cast(ch))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if cache is not None:
        df.to_parquet(cache, index=False)
    return df


class NormalEquations:
    """Accumulates X'X and X'y batch by batch -> exact least-squares solution without keeping the data in RAM.
    Used only to check that Gradient Descent reaches (almost) the best possible linear model."""

    def __init__(self, n_features):
        self.A = np.zeros((n_features + 1, n_features + 1)); self.c = np.zeros(n_features + 1); self.n = 0

    def add(self, X, y):
        Xb = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype="float64")])
        self.A += Xb.T @ Xb; self.c += Xb.T @ np.asarray(y, dtype="float64"); self.n += len(y)

    def solve(self):
        w = np.linalg.lstsq(self.A, self.c, rcond=None)[0]
        return float(w[0]), w[1:]


def formula_text(intercept, coefs, names, order=None, nd=4) -> str:
    idx = list(range(len(names))) if order is None else list(order)
    lines = [f"log1p(Adjusted_Demand) = {intercept:+.{nd}f}"]
    for i in idx:
        lines.append(f"                        {coefs[i]:+.{nd}f} * {names[i]}")
    lines.append("Adjusted_Demand = exp( log1p(Adjusted_Demand) ) - 1")
    return "\n".join(lines)


def _r2(y, p):
    sst = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan


def product_report(val: pd.DataFrame) -> pd.DataFrame:
    """One row per product: actual vs predicted demand on the validation week."""
    rows = []
    for pid, g in val.groupby("Product_ID"):
        rows.append(dict(Product_ID=int(pid), n_rows=len(g),
                         Actual_Mean=g["Adjusted_Demand_Units"].mean(), Predicted_Mean=g["Predicted_Demand"].mean(),
                         Actual_Total=g["Adjusted_Demand_Units"].sum(), Predicted_Total=g["Predicted_Demand"].sum(),
                         MAE_units=float(np.mean(np.abs(g["Error"]))),
                         R2_log=_r2(g["y_log"].to_numpy(), g["pred_log"].to_numpy())))
    out = pd.DataFrame(rows)
    out["Error_%"] = 100 * (out["Predicted_Total"] - out["Actual_Total"]) / out["Actual_Total"].replace(0, np.nan)
    return out.sort_values("Actual_Total", ascending=False).reset_index(drop=True)


class EDAAccumulator:
    """Exact exploratory statistics over ALL rows without ever holding them in memory together.
    Feed it one weekly DataFrame at a time with add(); read the results afterwards."""

    NUM = ["Sales_Units_Today", "Sales_Today", "Returns_Units_Next", "Returns_Next", "Adjusted_Demand_Units"]
    T = "Adjusted_Demand_Units"

    def __init__(self):
        self.n = 0; self.cols = None
        self.nulls = self.neg = self.zeros = self.vc = self.wk = self.ch = self.pw = None
        k = len(self.NUM)
        self.S, self.SS = np.zeros(k), np.zeros((k, k))
        self.mn, self.mx = np.full(k, np.inf), np.full(k, -np.inf)
        self.ps = {}

    @staticmethod
    def _acc(a, b):
        return b if a is None else a.add(b, fill_value=0)

    def _power_sums(self, col, tr, x):
        x = np.nan_to_num(x)
        v = np.array([len(x), x.sum(), (x ** 2).sum(), (x ** 3).sum(), (x ** 4).sum()])
        self.ps[(col, tr)] = self.ps.get((col, tr), 0) + v

    def add(self, d: pd.DataFrame):
        self.n += len(d); self.cols = list(d.columns)
        num = d[self.NUM]
        self.nulls = self._acc(self.nulls, d.isna().sum())
        self.neg = self._acc(self.neg, (num < 0).sum())
        self.zeros = self._acc(self.zeros, (num == 0).sum())
        self.vc = self._acc(self.vc, d.groupby(["Product_ID", self.T]).size())
        self.wk = self._acc(self.wk, d.groupby("Week")[self.T].agg(["sum", "size"]))
        self.ch = self._acc(self.ch, d.groupby("Channel_ID")[self.T].agg(["sum", "size"]))
        self.pw = self._acc(self.pw, d.groupby(["Product_ID", "Week"])[self.T].agg(["size", "sum"]))
        A = num.to_numpy("float64"); A0 = np.nan_to_num(A)
        self.S += A0.sum(axis=0); self.SS += A0.T @ A0
        self.mn = np.minimum(self.mn, np.nanmin(A, axis=0)); self.mx = np.maximum(self.mx, np.nanmax(A, axis=0))
        for j, c in enumerate(self.NUM):
            x = np.clip(A0[:, j], 0, None)
            self._power_sums(c, "raw", x); self._power_sums(c, "log1p", np.log1p(x))
        y = np.clip(A0[:, self.NUM.index(self.T)], 0, None)
        self._power_sums(self.T, "square root", np.sqrt(y)); self._power_sums(self.T, "cube root", np.cbrt(y))

    # ---------- results ----------
    def quality(self):
        q = pd.DataFrame(index=self.cols)
        q["NULLs"] = self.nulls.reindex(self.cols).fillna(0).astype(int)
        q["negative values"] = self.neg.reindex(self.cols).fillna(0).astype(int)
        q["zeros"] = self.zeros.reindex(self.cols).fillna(0).astype(int)
        return q

    def describe(self):
        mean = self.S / self.n
        std = np.sqrt(np.maximum(np.diag(self.SS) / self.n - mean ** 2, 0))
        return pd.DataFrame({"mean": mean, "std": std, "min": self.mn, "max": self.mx}, index=self.NUM)

    def corr(self):
        mean = self.S / self.n
        cov = self.SS / self.n - np.outer(mean, mean)
        sd = np.sqrt(np.diag(cov))
        return pd.DataFrame(cov / np.outer(sd, sd), index=self.NUM, columns=self.NUM)

    def target_hist(self):
        """Counts of each demand value (all products)."""
        return self.vc.groupby(level=1).sum().sort_index()

    def target_quantiles(self, qs=(0.5, 0.9, 0.99)):
        h = self.target_hist(); cum = h.cumsum() / h.sum()
        return {q: float(h.index[np.searchsorted(cum.to_numpy(), q)]) for q in qs}

    def product_profile(self):
        rows = []
        v_ = self.vc.reset_index(); v_.columns = ["Product_ID", "value", "count"]
        for pid, g in v_.groupby("Product_ID"):
            g = g.sort_values("value"); v = g["value"].to_numpy(float); c = g["count"].to_numpy(float); n = c.sum()
            mean = (v * c).sum() / n; var = ((v - mean) ** 2 * c).sum() / n
            skew = ((v - mean) ** 3 * c).sum() / n / var ** 1.5 if var > 0 else np.nan
            cum = np.cumsum(c) / n
            rows.append(dict(Product_ID=int(pid), rows=int(n), mean=mean, median=v[np.searchsorted(cum, .5)],
                             std=np.sqrt(var), p99=v[np.searchsorted(cum, .99)], max=v.max(),
                             zero_share=c[v == 0].sum() / n, skew=skew))
        return pd.DataFrame(rows).set_index("Product_ID").sort_values("rows", ascending=False)

    @staticmethod
    def _skew_kurt(v):
        n, s1, s2, s3, s4 = v; m = s1 / n; var = s2 / n - m ** 2
        c3 = s3 / n - 3 * m * s2 / n + 2 * m ** 3
        c4 = s4 / n - 4 * m * s3 / n + 6 * m ** 2 * s2 / n - 3 * m ** 4
        return (c3 / var ** 1.5, c4 / var ** 2 - 3) if var > 0 else (np.nan, np.nan)

    def skew_target(self):
        out = {tr: self._skew_kurt(self.ps[(self.T, tr)]) for tr in ["raw", "square root", "cube root", "log1p"]}
        return pd.DataFrame(out, index=["skewness", "excess kurtosis"]).T

    def skew_columns(self):
        return pd.DataFrame({"skew (raw)": {c: self._skew_kurt(self.ps[(c, "raw")])[0] for c in self.NUM},
                             "skew (log1p)": {c: self._skew_kurt(self.ps[(c, "log1p")])[0] for c in self.NUM}})

    def weekly(self):
        return self.wk.rename(columns={"size": "rows"})

    def channel(self):
        c = self.ch.copy(); c["mean"] = c["sum"] / c["size"]; return c.rename(columns={"size": "rows"})

    def product_week(self):
        d = self.pw.reset_index().rename(columns={"size": "n_rows", "sum": "total"})
        d["mean"] = d["total"] / d["n_rows"]
        return d[["Product_ID", "Week", "n_rows", "mean", "total"]]


class EDAAccumulator:
    """Exact exploratory statistics over ALL rows without ever holding them in memory together.
    Feed it one weekly DataFrame at a time with add(); read the results afterwards."""

    NUM = ["Sales_Units_Today", "Sales_Today", "Returns_Units_Next", "Returns_Next", "Adjusted_Demand_Units"]
    T = "Adjusted_Demand_Units"

    def __init__(self):
        self.n = 0; self.cols = None
        self.nulls = self.neg = self.zeros = self.vc = self.wk = self.ch = self.pw = None
        k = len(self.NUM)
        self.S, self.SS = np.zeros(k), np.zeros((k, k))
        self.mn, self.mx = np.full(k, np.inf), np.full(k, -np.inf)
        self.ps = {}

    @staticmethod
    def _acc(a, b):
        return b if a is None else a.add(b, fill_value=0)

    def _power_sums(self, col, tr, x):
        x = np.nan_to_num(x)
        v = np.array([len(x), x.sum(), (x ** 2).sum(), (x ** 3).sum(), (x ** 4).sum()])
        self.ps[(col, tr)] = self.ps.get((col, tr), 0) + v

    def add(self, d: pd.DataFrame):
        self.n += len(d); self.cols = list(d.columns)
        num = d[self.NUM]
        self.nulls = self._acc(self.nulls, d.isna().sum())
        self.neg = self._acc(self.neg, (num < 0).sum())
        self.zeros = self._acc(self.zeros, (num == 0).sum())
        self.vc = self._acc(self.vc, d.groupby(["Product_ID", self.T]).size())
        self.wk = self._acc(self.wk, d.groupby("Week")[self.T].agg(["sum", "size"]))
        self.ch = self._acc(self.ch, d.groupby("Channel_ID")[self.T].agg(["sum", "size"]))
        self.pw = self._acc(self.pw, d.groupby(["Product_ID", "Week"])[self.T].agg(["size", "sum"]))
        A = num.to_numpy("float64"); A0 = np.nan_to_num(A)
        self.S += A0.sum(axis=0); self.SS += A0.T @ A0
        self.mn = np.minimum(self.mn, np.nanmin(A, axis=0)); self.mx = np.maximum(self.mx, np.nanmax(A, axis=0))
        for j, c in enumerate(self.NUM):
            x = np.clip(A0[:, j], 0, None)
            self._power_sums(c, "raw", x); self._power_sums(c, "log1p", np.log1p(x))
        y = np.clip(A0[:, self.NUM.index(self.T)], 0, None)
        self._power_sums(self.T, "square root", np.sqrt(y)); self._power_sums(self.T, "cube root", np.cbrt(y))

    # ---------- results ----------
    def quality(self):
        q = pd.DataFrame(index=self.cols)
        q["NULLs"] = self.nulls.reindex(self.cols).fillna(0).astype(int)
        q["negative values"] = self.neg.reindex(self.cols).fillna(0).astype(int)
        q["zeros"] = self.zeros.reindex(self.cols).fillna(0).astype(int)
        return q

    def describe(self):
        mean = self.S / self.n
        std = np.sqrt(np.maximum(np.diag(self.SS) / self.n - mean ** 2, 0))
        return pd.DataFrame({"mean": mean, "std": std, "min": self.mn, "max": self.mx}, index=self.NUM)

    def corr(self):
        mean = self.S / self.n
        cov = self.SS / self.n - np.outer(mean, mean)
        sd = np.sqrt(np.diag(cov))
        return pd.DataFrame(cov / np.outer(sd, sd), index=self.NUM, columns=self.NUM)

    def target_hist(self):
        """Counts of each demand value (all products)."""
        return self.vc.groupby(level=1).sum().sort_index()

    def target_quantiles(self, qs=(0.5, 0.9, 0.99)):
        h = self.target_hist(); cum = h.cumsum() / h.sum()
        return {q: float(h.index[np.searchsorted(cum.to_numpy(), q)]) for q in qs}

    def product_profile(self):
        rows = []
        v_ = self.vc.reset_index(); v_.columns = ["Product_ID", "value", "count"]
        for pid, g in v_.groupby("Product_ID"):
            g = g.sort_values("value"); v = g["value"].to_numpy(float); c = g["count"].to_numpy(float); n = c.sum()
            mean = (v * c).sum() / n; var = ((v - mean) ** 2 * c).sum() / n
            skew = ((v - mean) ** 3 * c).sum() / n / var ** 1.5 if var > 0 else np.nan
            cum = np.cumsum(c) / n
            rows.append(dict(Product_ID=int(pid), rows=int(n), mean=mean, median=v[np.searchsorted(cum, .5)],
                             std=np.sqrt(var), p99=v[np.searchsorted(cum, .99)], max=v.max(),
                             zero_share=c[v == 0].sum() / n, skew=skew))
        return pd.DataFrame(rows).set_index("Product_ID").sort_values("rows", ascending=False)

    @staticmethod
    def _skew_kurt(v):
        n, s1, s2, s3, s4 = v; m = s1 / n; var = s2 / n - m ** 2
        c3 = s3 / n - 3 * m * s2 / n + 2 * m ** 3
        c4 = s4 / n - 4 * m * s3 / n + 6 * m ** 2 * s2 / n - 3 * m ** 4
        return (c3 / var ** 1.5, c4 / var ** 2 - 3) if var > 0 else (np.nan, np.nan)

    def skew_target(self):
        out = {tr: self._skew_kurt(self.ps[(self.T, tr)]) for tr in ["raw", "square root", "cube root", "log1p"]}
        return pd.DataFrame(out, index=["skewness", "excess kurtosis"]).T

    def skew_columns(self):
        return pd.DataFrame({"skew (raw)": {c: self._skew_kurt(self.ps[(c, "raw")])[0] for c in self.NUM},
                             "skew (log1p)": {c: self._skew_kurt(self.ps[(c, "log1p")])[0] for c in self.NUM}})

    def weekly(self):
        return self.wk.rename(columns={"size": "rows"})

    def channel(self):
        c = self.ch.copy(); c["mean"] = c["sum"] / c["size"]; return c.rename(columns={"size": "rows"})

    def product_week(self):
        d = self.pw.reset_index().rename(columns={"size": "n_rows", "sum": "total"})
        d["mean"] = d["total"] / d["n_rows"]
        return d[["Product_ID", "Week", "n_rows", "mean", "total"]]


# ======================================================================================================
#  Feature selection from accumulated sums (exact least squares for ANY subset of features, no re-reading data)
# ======================================================================================================
class GramAccumulator:
    """Accumulates Z'Z for Z = [1, X, y] chunk by chunk, plus power sums for skewness.
    From this one small matrix we can fit the exact least-squares model of ANY subset of features and read its
    R² and MSE, on the training rows (fit) and on other rows (validation) - without touching the data again."""

    def __init__(self, names):
        self.names = list(names); p = len(self.names)
        self.G = np.zeros((p + 2, p + 2)); self.n = 0
        self.pw = np.zeros((3, p + 1))                       # sums of x, x², x³ for every feature and for y

    def add(self, X, y):
        X = np.asarray(X, dtype="float64"); y = np.asarray(y, dtype="float64")
        Z = np.hstack([np.ones((len(X), 1)), X, y[:, None]])
        self.G += Z.T @ Z; self.n += len(y)
        W = Z[:, 1:]
        self.pw[0] += W.sum(0); self.pw[1] += (W ** 2).sum(0); self.pw[2] += (W ** 3).sum(0)

    # ---- helpers --------------------------------------------------------------------------------
    def _idx(self, feats):
        return [0] + [1 + self.names.index(f) for f in feats]

    def moments(self):
        mean = self.pw[0] / self.n; var = self.pw[1] / self.n - mean ** 2
        m3 = self.pw[2] / self.n - 3 * mean * self.pw[1] / self.n + 2 * mean ** 3
        skew = np.where(var > 1e-12, m3 / np.power(np.maximum(var, 1e-12), 1.5), np.nan)
        return pd.DataFrame({"mean": mean, "std": np.sqrt(np.maximum(var, 0)), "skewness": skew}, index=self.names + ["y"])

    def corr(self):
        W = self.G[1:, 1:]; m = self.pw[0] / self.n
        cov = W / self.n - np.outer(m, m); sd = np.sqrt(np.maximum(np.diag(cov), 1e-18))
        return pd.DataFrame(cov / np.outer(sd, sd), index=self.names + ["y"], columns=self.names + ["y"])

    def fit(self, feats):
        """Exact least squares for the given features -> (intercept, coefficients)."""
        i = self._idx(feats); A = self.G[np.ix_(i, i)]; c = self.G[i, -1]
        b = np.linalg.lstsq(A, c, rcond=None)[0]
        return float(b[0]), b[1:]

    def score(self, feats, intercept, coef):
        """R² and MSE of a given equation on the rows accumulated here."""
        i = self._idx(feats); beta = np.r_[intercept, coef]
        A = self.G[np.ix_(i, i)]; c = self.G[i, -1]; yy = self.G[-1, -1]
        sse = float(yy - 2 * beta @ c + beta @ A @ beta)
        sst = float(yy - self.G[0, -1] ** 2 / self.n)
        return {"R2": 1 - sse / sst, "MSE": sse / self.n}


def reset_cache_if_table_changed(cache_dir, tag, table, counts) -> bool:
    """Delete the parquet cache of `tag` when the SQL table has changed (other products / other row counts), so that
    old cached weeks are never reused by mistake. `counts` = the DataFrame returned by count_rows(). Returns True if cleared."""
    import hashlib
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    sig = hashlib.md5((table + "|" + counts.sort_values(list(counts.columns)).to_csv(index=False)).encode()).hexdigest()
    f = cache_dir / f"{tag}_signature.txt"
    changed = not f.exists() or f.read_text().strip() != sig
    if changed:
        for p in cache_dir.glob(f"{tag}_week*.parquet"):
            p.unlink()
        f.write_text(sig)
    return changed
