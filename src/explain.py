"""explain.py – SHAP-viz + Chain-of-Thought log

CLI-használat:
python -m src.explain \
       --selected_csv output/…/live_selected.csv \
       --weights      output/…/weights.json \
       --out          output/…/

Funkciók
--------
* XGBoostRegressor (lightweight) illesztése a faktor-feature-ekből → portfólió-súlyok.
* SHAP `summary_plot` PNG-t generál, amely megmutatja, mely inputok
  befolyásolják leginkább a súlyokat.
* LLM-lánc (prompt-ek) naplóját – ha a `cot_log.json` létezik ugyanabban
  a könyvtárban, kombinálja; ha nincs, létrehoz egy minimális logot.

Követelmények: xgboost, shap, pandas, matplotlib (a requirements.txt-ben már szerepelnek).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List
from datetime import datetime, timezone

import pandas as pd
import shap
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s", force=True)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def _load_data(csv_path: Path, weights_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for modelling."""

    df = pd.read_csv(csv_path)
    with weights_path.open() as fp:
        w = json.load(fp)

    w_series = pd.Series(w, name="weight").astype(float)
    df = df.merge(w_series, left_on="symbol", right_index=True, how="inner")

    # Use only numeric feature columns (exclude symbol, Close, etc.)
    drop_cols: List[str] = [
        "symbol",
        "Close",
        "sector",  # esetleg létezik
        "weight",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != "object"]

    X = df[feature_cols].fillna(0)
    y = df["weight"].fillna(0)
    return X, y


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def explain(selected_csv: Path, weights_path: Path, out_dir: Path) -> None:
    X, y = _load_data(selected_csv, weights_path)
    if len(X) < 3:
        LOGGER.error("Need at least 3 samples for SHAP plot, got %d", len(X))
        sys.exit(1)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    out_dir.mkdir(parents=True, exist_ok=True)

    # SHAP summary PNG
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, show=False)
    shap_fp = out_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_fp, dpi=300)
    plt.close()
    LOGGER.info("SHAP summary saved → %s", shap_fp)

    # Chain-of-Thought log (append or create)
    cot_fp = out_dir / "cot_log.json"
    cot_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "weights_explainer",
        "note": "Trained XGBRegressor on factor features; generated SHAP summary PNG.",
    }
    if cot_fp.exists():
        with cot_fp.open("r", encoding="utf-8") as f:
            cot_data = json.load(f)
            if not isinstance(cot_data, list):  # edge-case sanity
                cot_data = [cot_data]
    else:
        cot_data = []
    cot_data.append(cot_entry)
    cot_fp.write_text(json.dumps(cot_data, indent=2))
    LOGGER.info("CoT log updated → %s", cot_fp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("SHAP + CoT explainer")
    p.add_argument("--selected_csv", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    explain(a.selected_csv, a.weights, a.out)
