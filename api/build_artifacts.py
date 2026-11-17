"""OPTIONAL (author only): build the small API artifacts from the full training outputs.

The training notebook saves a 200 MB history file (all 22 candidate features). The final equation needs only four
features, so this script keeps just the lookup tables those four features use and writes:
    artifacts/model.npz              weights, standardisation (mean / std) and feature names
    artifacts/history_compact.npz    product table, customer table, pair table (average of the last 3 orders)
    artifacts/model_metadata.json    model_id, example request and its reference forecast (computed with the FULL pipeline)
Usage:  python build_artifacts.py <project_folder_with_bimbo_gd> <outputs_final_folder>
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

project, final = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(project))
from bimbo_gd.gradient_descent import LinearRegressionGD  # noqa: E402

out = Path(__file__).resolve().parent / "artifacts"
out.mkdir(exist_ok=True)

model, feats = LinearRegressionGD.load(final / "model_final.npz")
raw = np.load(final / "model_final.npz", allow_pickle=True)
with open(final / "history_stats_final.pkl", "rb") as f:
    builder = pickle.load(f)
assert list(feats) == ["mean3_cust_prod", "te_product", "is_new_cust_prod", "te_customer"] or len(feats) == 4, feats
print("features:", list(feats), "| history weeks:", builder.weeks_seen)

# ---- compact lookup tables -------------------------------------------------------------------------------
prod = builder.tables["product"][["Product_ID", "te_product"]]
cust = builder.tables["customer"][["Customer_ID", "te_customer"]]
P = builder.pairs
last = P["last_cust_prod"].to_numpy("float64")
prev = np.where(np.isnan(P["prev_cust_prod"]), last, P["prev_cust_prod"].to_numpy("float64"))
prev2 = np.where(np.isnan(P["prev2_cust_prod"]), prev, P["prev2_cust_prod"].to_numpy("float64"))
mean3 = ((last + prev + prev2) / 3.0).astype("float32")
order = np.lexsort((P["Product_ID"].to_numpy(), P["Customer_ID"].to_numpy()))
np.savez_compressed(
    out / "history_compact.npz",
    gm=np.float64(builder.gm),
    product_id=prod["Product_ID"].to_numpy("int32"), te_product=prod["te_product"].to_numpy("float32"),
    customer_id=cust["Customer_ID"].to_numpy("int32"), te_customer=cust["te_customer"].to_numpy("float32"),
    pair_customer=P["Customer_ID"].to_numpy("int32")[order], pair_product=P["Product_ID"].to_numpy("int32")[order],
    pair_mean3=mean3[order],
    weeks=np.array(builder.weeks_seen, dtype="int32"))
np.savez(out / "model.npz", w=raw["w"], b=raw["b"], mu=raw["mu"], sigma=raw["sigma"], features=np.array(list(feats)))

# ---- example + reference forecast computed with the ORIGINAL full pipeline ---------------------------------
import pandas as pd  # noqa: E402
ex = P.iloc[len(P) // 2]
rows = pd.DataFrame({"Week": [10], "Agency_ID": [0], "Channel_ID": [0], "Route_SAK": [0],
                     "Customer_ID": [int(ex["Customer_ID"])], "Product_ID": [int(ex["Product_ID"])]})
X = builder.transform(rows)[list(feats)]
pred_log = float(model.predict(X.to_numpy())[0])
mid = hashlib.sha256((out / "model.npz").read_bytes() + (out / "history_compact.npz").read_bytes()).hexdigest()[:12]
meta = {
    "model_id": f"bimbo-gd-linear-{mid}",
    "task": "regression: forecast log1p(Adjusted_Demand_Units) of a customer-product pair for the next week",
    "algorithm": "multiple linear regression trained with mini-batch Gradient Descent (own NumPy implementation)",
    "features": list(feats),
    "history_weeks": [int(w) for w in builder.weeks_seen],
    "example": {
        "request": {"customer_id": int(ex["Customer_ID"]), "product_id": int(ex["Product_ID"])},
        "predicted_log_demand": pred_log,
        "predicted_units": float(max(np.expm1(pred_log), 0.0)),
        "features": {k: float(X.iloc[0][k]) for k in feats},
    },
}
(out / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
for p in out.iterdir():
    print(p.name, f"{p.stat().st_size / 1e6:.2f} MB")
