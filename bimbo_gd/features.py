"""HISTORY-based feature engineering (no data leakage).

The test set only has: Week, Agency, Channel, Route, Customer, Product.
So Sales_Units_Today, Returns... of the week being forecast must NOT be used
directly. Instead we aggregate statistics from PREVIOUS weeks (history).

HistoricalFeatureBuilder supports incremental updates (update) batch by batch / week by week,
so the whole dataset never has to be loaded into RAM.
"""
import numpy as np
import pandas as pd

from .config import TARGET

GROUPS = {
    "product": ["Product_ID"],
    "customer": ["Customer_ID"],
    "agency": ["Agency_ID"],
    "route": ["Route_SAK"],
    "channel": ["Channel_ID"],
    "cust_prod": ["Customer_ID", "Product_ID"],
}

FEATURES = (
    [f"te_{g}" for g in GROUPS]            # smoothed historical mean of log-demand per group
    + [f"logcnt_{g}" for g in GROUPS]      # log(1 + historical observations / weeks) ~ popularity, stable over time
    + ["last_cust_prod",                   # most recent log-demand of the customer-product pair (lag 1)
       "prev_cust_prod",                   # the observation before that (lag 2)
       "mean3_cust_prod",                  # average of the last three observations of the pair
       "gap_cust_prod",                    # log(1 + weeks since the pair was last seen)
       "ret_cust_prod",                    # sqrt of the pair's historical return rate (high = over-stocked store)
       "log_hist_price", "hist_return_rate",
       "is_new_customer", "is_new_product", "is_new_cust_prod"]
)

_HIST_COLS = ["Week", "Agency_ID", "Channel_ID", "Route_SAK", "Customer_ID", "Product_ID",
              "Sales_Units_Today", "Sales_Today", "Returns_Units_Next"]


def log_target(s: pd.Series) -> np.ndarray:
    """log1p(demand) - also the scale of RMSLE, the original competition metric."""
    return np.log1p(s.clip(lower=0).to_numpy(dtype="float64"))


class HistoricalFeatureBuilder:
    def __init__(self, smoothing: float = 5.0):
        self.m = smoothing
        self.reset()

    def reset(self):
        self.n_rows, self.ysum, self.gm = 0, 0.0, 0.0
        self.sums, self.pairs, self.prod_sums = {}, None, None
        self.tables, self.prod = {}, None
        self.weeks_seen = []
        return self

    # ---------- learn statistics (call repeatedly, one batch at a time, in week order) ----------
    def update(self, hist: pd.DataFrame) -> "HistoricalFeatureBuilder":
        d = hist[_HIST_COLS].copy()
        d["y"] = log_target(hist[TARGET])
        self.n_rows += len(d); self.ysum += float(d["y"].sum()); self.gm = self.ysum / self.n_rows
        self.weeks_seen += sorted(set(d["Week"].unique().tolist()) - set(self.weeks_seen))

        for name, keys in GROUPS.items():
            g = d.groupby(keys, sort=False)["y"].agg(["sum", "count"]).reset_index()
            if name in self.sums:
                g = pd.concat([self.sums[name], g], ignore_index=True) \
                      .groupby(keys, sort=False)[["sum", "count"]].sum().reset_index()
            self.sums[name] = g

        self._update_pairs(d)

        p = d.groupby("Product_ID", sort=False)[["Sales_Today", "Sales_Units_Today", "Returns_Units_Next"]] \
             .sum().reset_index()
        if self.prod_sums is not None:
            p = pd.concat([self.prod_sums, p], ignore_index=True) \
                  .groupby("Product_ID", sort=False).sum().reset_index()
        self.prod_sums = p
        self._refresh()
        return self

    _PAIR_COLS = ["last_cust_prod", "prev_cust_prod", "prev2_cust_prod", "last_week", "ret_sum", "unit_sum"]

    def _update_pairs(self, d: pd.DataFrame):
        """Per customer-product pair keep the last 3 observed log-demands, the week last seen and the return history."""
        ck = GROUPS["cust_prod"]
        per = d.groupby(ck + ["Week"], sort=False).agg(ny=("y", "mean"), nr=("Returns_Units_Next", "sum"),
                                                      nu=("Sales_Units_Today", "sum")).reset_index()
        P = self.pairs
        for wk, g in per.groupby("Week", sort=True):
            g = g.drop(columns="Week")
            if P is None:
                P = g[ck].copy()
                P["last_cust_prod"], P["prev_cust_prod"], P["prev2_cust_prod"] = g["ny"].to_numpy(), np.nan, np.nan
                P["last_week"], P["ret_sum"], P["unit_sum"] = float(wk), g["nr"].to_numpy(float), g["nu"].to_numpy(float)
                continue
            M = P.merge(g, on=ck, how="outer"); hit = M["ny"].notna().to_numpy()
            M["prev2_cust_prod"] = np.where(hit, M["prev_cust_prod"], M["prev2_cust_prod"])
            M["prev_cust_prod"] = np.where(hit, M["last_cust_prod"], M["prev_cust_prod"])
            M["last_cust_prod"] = np.where(hit, M["ny"], M["last_cust_prod"])
            M["last_week"] = np.where(hit, float(wk), M["last_week"])
            M["ret_sum"] = M["ret_sum"].fillna(0) + M["nr"].fillna(0)
            M["unit_sum"] = M["unit_sum"].fillna(0) + M["nu"].fillna(0)
            P = M[ck + self._PAIR_COLS]
        self.pairs = P.reset_index(drop=True)

    def fit(self, hist: pd.DataFrame) -> "HistoricalFeatureBuilder":
        """Learn from scratch on a single DataFrame (used by the small-sample pipeline)."""
        return self.reset().update(hist)

    def _refresh(self):
        for name, keys in GROUPS.items():
            g = self.sums[name].copy()
            g[f"te_{name}"] = (g["sum"] + self.m * self.gm) / (g["count"] + self.m)
            # average count per week -> stable as history grows (avoids drift between batches)
            g[f"logcnt_{name}"] = np.log1p(g["count"] / max(1, len(self.weeks_seen)))
            self.tables[name] = g[keys + [f"te_{name}", f"logcnt_{name}"]]
        p = self.prod_sums.copy()
        units = p["Sales_Units_Today"].replace(0, np.nan)
        p["log_hist_price"] = np.log1p((p["Sales_Today"] / units).clip(lower=0))
        p["hist_return_rate"] = (p["Returns_Units_Next"] / units).clip(0, 1)
        self.prod = p[["Product_ID", "log_hist_price", "hist_return_rate"]]
        self.med_price = float(self.prod["log_hist_price"].median())
        self.med_ret = float(self.prod["hist_return_rate"].median())

    def slim(self):
        """Shrink the builder before saving: drop the raw sums (only needed for further updates) and store the
        learned statistics as float32. After slim() the builder can transform() but no longer update()."""
        self.sums, self.prod_sums = {}, None
        for name in self.tables:
            t = self.tables[name]
            for c in t.columns:
                if c.startswith(("te_", "logcnt_")):
                    t[c] = t[c].astype("float32")
        for c in self._PAIR_COLS:
            self.pairs[c] = self.pairs[c].astype("float32")
        return self

    # ---------- feature generation ----------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[["Agency_ID", "Channel_ID", "Route_SAK", "Customer_ID", "Product_ID"]].copy()
        for name, keys in GROUPS.items():
            X = X.merge(self.tables[name], on=keys, how="left")
        X = X.merge(self.pairs, on=GROUPS["cust_prod"], how="left")
        X = X.merge(self.prod, on="Product_ID", how="left")
        wk_now = df["Week"].to_numpy(dtype="float64")

        X["is_new_customer"] = X["logcnt_customer"].isna().astype("float64")
        X["is_new_product"] = X["logcnt_product"].isna().astype("float64")
        X["is_new_cust_prod"] = X["logcnt_cust_prod"].isna().astype("float64")

        for name in ["product", "customer", "agency", "route", "channel"]:
            X[f"te_{name}"] = X[f"te_{name}"].fillna(self.gm)
        # unseen customer-product pair -> fall back to the product's average level
        X["te_cust_prod"] = X["te_cust_prod"].fillna(X["te_product"])
        X["last_cust_prod"] = X["last_cust_prod"].fillna(X["te_cust_prod"])
        X["prev_cust_prod"] = X["prev_cust_prod"].fillna(X["last_cust_prod"])
        X["prev2_cust_prod"] = X["prev2_cust_prod"].fillna(X["prev_cust_prod"])
        X["mean3_cust_prod"] = (X["last_cust_prod"] + X["prev_cust_prod"] + X["prev2_cust_prod"]) / 3.0
        X["gap_cust_prod"] = np.log1p(np.clip(wk_now - np.where(X["last_week"].isna().to_numpy(), wk_now, X["last_week"].to_numpy(dtype="float64")), 0, None))
        for g in GROUPS:
            X[f"logcnt_{g}"] = X[f"logcnt_{g}"].fillna(0.0)
        X["log_hist_price"] = X["log_hist_price"].fillna(self.med_price)
        X["hist_return_rate"] = X["hist_return_rate"].fillna(self.med_ret)
        k = 5.0                                    # shrink the pair's return rate towards the product's rate
        X["ret_cust_prod"] = np.sqrt(((X["ret_sum"].fillna(0) + k * X["hist_return_rate"]) / (X["unit_sum"].fillna(0) + k)).clip(0, 1))   # sqrt tames the right skew
        return X[FEATURES].astype("float64")


def split_by_week(df: pd.DataFrame, weeks: list[int]) -> pd.DataFrame:
    return df[df["Week"].isin(weeks)].reset_index(drop=True)


# ======================================================================================================
#  Feature dictionary: what every feature means and how it is calculated (used by notebook 04 and the P1 document)
#  type: "average" = smoothed historical mean of log1p(demand); "count" = how often something occurs (already put on a
#  per-week log scale, so it is NOT log-transformed a second time); "lag" = earlier demand of the same pair;
#  "rate" = share between 0 and 1; "volume" = continuous positive amount; "flag" = 0/1 indicator.
# ======================================================================================================
_G = "G = the overall average of log1p(demand) in the history; m = 5 (smoothing weight)"
FEATURE_INFO = {
    "te_product":   ("average", "Typical demand level of the product", "(sum of log1p(demand) of the product's earlier rows + m*G) / (number of those rows + m)"),
    "te_customer":  ("average", "Typical demand level of the store", "same formula, over the store's earlier rows"),
    "te_agency":    ("average", "Typical demand level of the depot", "same formula, over the depot's earlier rows"),
    "te_route":     ("average", "Typical demand level of the delivery route", "same formula, over the route's earlier rows"),
    "te_channel":   ("average", "Typical demand level of the sales channel", "same formula, over the channel's earlier rows"),
    "te_cust_prod": ("average", "Typical demand of this store for this product", "same formula, over the earlier rows of the store-product pair; if the pair is new, the product's te_product is used"),
    "logcnt_product":   ("count", "How often the product is ordered", "log(1 + earlier order lines of the product / weeks of history)"),
    "logcnt_customer":  ("count", "How often the store orders", "log(1 + earlier order lines of the store / weeks of history)"),
    "logcnt_agency":    ("count", "How busy the depot is", "log(1 + earlier order lines of the depot / weeks of history)"),
    "logcnt_route":     ("count", "How busy the route is", "log(1 + earlier order lines of the route / weeks of history)"),
    "logcnt_channel":   ("count", "How busy the sales channel is", "log(1 + earlier order lines of the channel / weeks of history)"),
    "logcnt_cust_prod": ("count", "How regularly the store orders this product", "log(1 + earlier order lines of the store-product pair / weeks of history)"),
    "last_cust_prod":   ("lag", "Most recent demand of the store for this product", "log1p(demand) in the latest earlier week in which the pair was ordered; if new, te_cust_prod"),
    "prev_cust_prod":   ("lag", "The order before the most recent one", "log1p(demand) of the second-latest earlier order of the pair; if missing, last_cust_prod"),
    "mean3_cust_prod":  ("lag", "Average of the last three orders", "mean of last_cust_prod, prev_cust_prod and the third-latest order of the pair"),
    "gap_cust_prod":    ("count", "Weeks since the store last ordered this product", "log(1 + (current week - week the pair was last seen)); 0 if never seen"),
    "ret_cust_prod":    ("rate", "Share of this store's product that came back unsold", "sqrt of (returned units + 5*p) / (sold units + 5), where p = the product's overall return rate; the square root reduces the right skew"),
    "log_hist_price":   ("volume", "Price level of the product", "log(1 + historical sales value in pesos / units sold) of the product"),
    "hist_return_rate": ("rate", "Share of the product that usually comes back unsold", "historical returned units / units sold of the product (0 to 1)"),
    "is_new_customer":  ("flag", "Store never seen before", "1 if the store has no earlier history, otherwise 0"),
    "is_new_product":   ("flag", "Product never seen before", "1 if the product has no earlier history, otherwise 0"),
    "is_new_cust_prod": ("flag", "Store has never ordered this product before", "1 if the store-product pair has no earlier history, otherwise 0"),
}
assert set(FEATURE_INFO) == set(FEATURES), "FEATURE_INFO must describe every feature"
