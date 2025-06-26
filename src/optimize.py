"""optimize.py – Black‑Litterman + Mean‑Variance  v2.0
Robusztus, "megállíthatatlan" portfólió‑súly generátor.
⟶ max Sharpe  ➜  min Vol  ➜  Inverse‑Variance  ➜  Egyenlő súlyok
Minden ágon ticker‑kulcsos JSON‑t ír, sosem áll le None‑nal.
Run example:
python -m src.optimize \
       --selected_csv output/2025-06-25/live_selected.csv \
       --out output/2025-06-25
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence, Tuple, Dict

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models
from pypfopt.black_litterman import market_implied_risk_aversion

from .fetch_data import get_prices_polygon

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s", force=True)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------

def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def _load_prices(symbols: Sequence[str], parquet: Path | None) -> pd.DataFrame:
    if parquet and parquet.exists():
        return pd.read_parquet(parquet)[symbols]
    LOGGER.info("Fetching 1-year prices via Polygon …")
    end, start = date.today(), date.today() - timedelta(days=365)
    return get_prices_polygon(symbols, start, end)

# ---------------------------------------------------------------------------
# core solver chain
# ---------------------------------------------------------------------------

def _build_ef(mu: pd.Series, cov: np.ndarray) -> EfficientFrontier:
    ef = EfficientFrontier(mu, cov, solver="ECOS_BB", weight_bounds=(0, 0.6))
    ef.add_objective(lambda w: 1e-4 * cp.sum_squares(w))  # L2 ridge
    return ef


def _solve_weights(mu: pd.Series, cov: np.ndarray) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Return (weights, (exp_ret, vol, sharpe)) – never fails."""

    for mode in ("sharpe", "minvol"):
        try:
            ef = _build_ef(mu, cov)
            if mode == "sharpe":
                ef.max_sharpe()
            else:
                ef.min_volatility()
            w = ef.clean_weights()
            if not any(np.isnan(v) for v in w.values()):
                return w, ef.portfolio_performance(verbose=False)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s optimisation failed → %s", mode, exc)

    # inverse‑variance fallback
    try:
        ivp = 1 / np.diag(cov)
        ivp /= ivp.sum()
        w_ivp = {sym: float(w) for sym, w in zip(mu.index, ivp)}
        ret = float(np.dot(ivp, mu))
        vol = float(np.sqrt(ivp @ cov @ ivp))
        sharpe = ret / vol if vol else 0.0
        LOGGER.warning("Using inverse‑variance weights fallback.")
        return w_ivp, (ret, vol, sharpe)
    except Exception as exc:  # végső mentőöv
        LOGGER.error("IVP failed (%s) – reverting to equal weights", exc)
        eq = np.repeat(1 / len(mu), len(mu))
        w_eq = {sym: float(w) for sym, w in zip(mu.index, eq)}
        ret = float(np.dot(eq, mu))
        vol = float(np.sqrt(eq @ cov @ eq))
        sharpe = ret / vol if vol else 0.0
        return w_eq, (ret, vol, sharpe)

# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def run(selected_csv: Path, prices_path: Path | None, tau: float, out_dir: Path) -> None:
    df = pd.read_csv(selected_csv)
    symbols = df["symbol"].tolist()

    # market caps → weights
    caps = df["marketCapTTM"].astype(float).ffill().to_numpy()
    w_mkt = caps / caps.sum()

    # prices & covariance
    prices = _load_prices(symbols, prices_path)
    Σ = risk_models.sample_cov(prices).values + 1e-6 * np.eye(len(symbols))

    # prior returns π = δ Σ w
    δ = market_implied_risk_aversion(prices)
    π = δ * Σ.dot(w_mkt)

    bl = BlackLittermanModel(
        Σ,
        P=np.eye(len(symbols)),
        Q=_z(df["factor_score"]).values,
        pi=π,
        omega=np.diag(np.full(len(symbols), tau)),
        tau=tau,
        w_market=w_mkt,
    )
    μ = pd.Series(bl.bl_returns(), index=symbols)  # ensure ticker index
    Σ_bl = bl.bl_cov().values + 1e-6 * np.eye(len(symbols))

    weights, (exp_ret, vol, sharpe) = _solve_weights(μ, Σ_bl)

    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.json").write_text(json.dumps(weights, indent=2))
    (out_dir / "frontier_perf.json").write_text(
        json.dumps({"exp_return": exp_ret, "volatility": vol, "sharpe": sharpe}, indent=2)
    )
    print(f"✓ Portfolio weights → {out_dir / 'weights.json'}  ({len(weights)} assets)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("BL‑MeanVariance optimiser")
    p.add_argument("--selected_csv", type=Path, required=True)
    p.add_argument("--prices", type=Path, default=None)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if not args.selected_csv.exists():
        LOGGER.error("selected_csv missing: %s", args.selected_csv)
        sys.exit(1)
    run(args.selected_csv, args.prices, args.tau, args.out)
