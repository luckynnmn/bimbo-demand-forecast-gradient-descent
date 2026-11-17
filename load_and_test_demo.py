"""Load-and-test demonstration of the saved, fitted model (P2 requirement).

Run in a FRESH process (no database needed):   python 06_load_and_test_demo.py

1. Loads the saved model + saved history statistics (nothing is re-trained).
2. Checks the loaded model reproduces the predictions stored at training time (PASS / FAIL).
3. Forecasts a few example (customer, product) pairs with the loaded artifacts.
"""
import pickle
import sys
import numpy as np
import pandas as pd

from bimbo_gd import config
from bimbo_gd.gradient_descent import LinearRegressionGD

for p in (config.FINAL_MODEL_PATH, config.FINAL_BUILDER_PATH, config.FINAL_REF_PATH):
    if not p.exists():
        sys.exit(f"Missing {p}. Run notebook 04_train_full_filtered.ipynb first.")

model, feats = LinearRegressionGD.load(config.FINAL_MODEL_PATH)
with open(config.FINAL_BUILDER_PATH, "rb") as f:
    builder = pickle.load(f)
w0, w = model.raw_coefficients()
print(f"Loaded model: {len(feats)} features | intercept {w0:+.4f} | history weeks {builder.weeks_seen}")

ref = np.load(config.FINAL_REF_PATH)
ok = np.allclose(model.predict(ref["X"]), ref["pred"])
print(f"Check 1 - same predictions as at training time on {len(ref['pred']):,} stored rows: {'PASS' if ok else 'FAIL'}")

# Example forecast: a new row of the kind found in the test table (IDs below are placeholders -
# unseen IDs simply fall back to the product average, which is the intended behaviour).
products = json_products = None
try:
    import json
    products = json.loads(config.FINAL_METRICS_PATH.read_text())["scope"]["products"]
except Exception:
    pass
pid = products[0] if products else 1240
rows = pd.DataFrame({"Week": [10] * 3, "Agency_ID": [1110] * 3, "Channel_ID": [1] * 3, "Route_SAK": [1001] * 3,
                     "Customer_ID": [397854, 397855, 999999999], "Product_ID": [pid] * 3})
X = builder.transform(rows)[feats]
units = np.expm1(model.predict(X.to_numpy())).clip(min=0)
print("\nCheck 2 - example forecasts (the third customer is unknown to the model, so the product average is used):")
print(pd.DataFrame({"Customer_ID": rows.Customer_ID, "Product_ID": rows.Product_ID, "forecast_units": units.round(2)}).to_string(index=False))
sys.exit(0 if ok else 1)
