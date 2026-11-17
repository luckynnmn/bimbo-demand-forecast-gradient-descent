"""Local forecast API for the frozen Grupo Bimbo model.

Start:   python -m uvicorn app:app --host 127.0.0.1 --port 8000
Check:   python smoke_check_api.py          (in a second terminal)
Docs:    http://127.0.0.1:8000/docs

GET  /health   -> {"status": "ready", "model_id": "..."}
POST /predict  -> forecast of next week's demand of one customer-product pair, e.g. {"customer_id": 1502792, "product_id": 1242}

The service needs only NumPy: no database and no training code. It loads the frozen equation (artifacts/model.npz) and the
history lookup tables (artifacts/history_compact.npz) and rebuilds the same four features as training.
"""
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
META = json.loads((ART / "model_metadata.json").read_text(encoding="utf-8"))

_m = np.load(ART / "model.npz", allow_pickle=False)
W, B, MU, SIGMA = _m["w"], float(_m["b"]), _m["mu"], _m["sigma"]
FEATURES = [str(f) for f in _m["features"]]

_h = np.load(ART / "history_compact.npz", allow_pickle=False)
GM = float(_h["gm"])
PRODUCT_ID, TE_PRODUCT = _h["product_id"], _h["te_product"]
CUSTOMER_ID, TE_CUSTOMER = _h["customer_id"], _h["te_customer"]
_PAIR_KEY = _h["pair_customer"].astype("int64") * 65536 + _h["pair_product"].astype("int64")   # sorted
PAIR_MEAN3 = _h["pair_mean3"]
_ORDER_P, _ORDER_C = np.argsort(PRODUCT_ID), np.argsort(CUSTOMER_ID)


def _lookup(ids_sorted_index, ids, values, key):
    pos = np.searchsorted(ids[ids_sorted_index], key)
    if pos < len(ids) and ids[ids_sorted_index][pos] == key:
        return float(values[ids_sorted_index][pos]), True
    return None, False


def build_features(customer_id: int, product_id: int) -> tuple[dict, dict]:
    """Same four features as training: te_product, te_customer, mean3_cust_prod, is_new_cust_prod."""
    te_p, known_p = _lookup(_ORDER_P, PRODUCT_ID, TE_PRODUCT, product_id)
    te_c, known_c = _lookup(_ORDER_C, CUSTOMER_ID, TE_CUSTOMER, customer_id)
    te_p = GM if te_p is None else te_p                      # unknown product -> overall average
    te_c = GM if te_c is None else te_c                      # unknown customer -> overall average
    key = customer_id * 65536 + product_id
    pos = int(np.searchsorted(_PAIR_KEY, key)) if 0 <= product_id < 65536 else len(_PAIR_KEY)
    known_pair = pos < len(_PAIR_KEY) and _PAIR_KEY[pos] == key
    mean3 = float(PAIR_MEAN3[pos]) if known_pair else te_p   # new pair -> the product's average level
    feats = {"mean3_cust_prod": mean3, "te_product": te_p, "te_customer": te_c,
             "is_new_cust_prod": 0.0 if known_pair else 1.0}
    return feats, {"known_customer": known_c, "known_product": known_p, "known_pair": bool(known_pair)}


def forecast(customer_id: int, product_id: int) -> dict:
    feats, known = build_features(customer_id, product_id)
    x = np.array([feats[name] for name in FEATURES], dtype="float64")
    log_pred = float(((x - MU) / SIGMA) @ W + B)
    return {"model_id": META["model_id"], "customer_id": customer_id, "product_id": product_id,
            "predicted_log_demand": log_pred, "predicted_units": float(max(np.expm1(log_pred), 0.0)),
            "features": feats, **known}


class PredictRequest(BaseModel):
    customer_id: int = Field(..., ge=0, description="Customer (store) ID, as in the Bimbo data")
    product_id: int = Field(..., ge=0, le=65535, description="Product ID (one of the 10 products used in the project)")


app = FastAPI(title="Grupo Bimbo demand forecast", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ready", "model_id": META["model_id"]}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    return forecast(req.customer_id, req.product_id)
