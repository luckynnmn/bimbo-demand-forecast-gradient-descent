"""Smoke-check the running local API with Python's standard library."""

import json
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    root = Path(__file__).resolve().parent
    metadata = json.loads(
        (root / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"),
    )
    with urlopen("http://127.0.0.1:8000/health", timeout=10) as response:
        health = json.load(response)
    if health != {"status": "ready", "model_id": metadata["model_id"]}:
        raise RuntimeError(f"Unexpected health response: {health}")
    request = Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps(metadata["example"]["request"]).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    if abs(result["predicted_log_demand"] - metadata["example"]["predicted_log_demand"]) > 1e-6:
        raise RuntimeError(f"Forecast differs from the supplied reference: {result}")
    if abs(result["predicted_units"] - metadata["example"]["predicted_units"]) > 1e-4:
        raise RuntimeError(f"Predicted units differ from the reference: {result}")
    print("PASS: health and forecast match the supplied reference.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
