# Forecast API (P2)

Serves the frozen Gradient Descent linear regression (4 features). Needs only Python, FastAPI, uvicorn and NumPy; no SQL Server.

## Run
```bash
cd api
pip install -r requirements-api.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Smoke check (second terminal, standard library only)
```bash
python smoke_check_api.py
```
Expected last line: `PASS: health and forecast match the supplied reference.`

## Manual test
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"customer_id":1502792,"product_id":1242}'
```

## Artifacts (`artifacts/`)
- `model.npz`: weights, intercept, standardization means/stds, feature names
- `history_compact.npz`: lookup tables for the 4 features (13 MB)
- `model_metadata.json`: model_id and the reference example used by the smoke check

Unknown stores/products get the overall average level; the response flags `known_customer`, `known_product`, `known_pair`.

## Rebuild (optional, needs the 203 MB `history_stats_final.pkl`, not in GitHub)
`python build_artifacts.py <project_folder> <outputs_final_folder>`
