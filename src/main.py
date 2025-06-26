"""main.py – CLI orchestrator  v3  (2025-06-25)
• Composite Fundamental  = z(−PE) + z(ROE) → 0-1
• Technical              = 20-napos RSI (0-100, semleges 50)
• Macro                  = szektor-β × 10Y-hozam Δ%
• NaN-robust momentum-szűrés kis univerzumnál
• CSV mindig kiíródik, INFO log sor-számmal
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .factors import (
    calc_four_factor,
    calc_roc,
    calc_fundamental_composite,
    calc_technical_rsi,
    calc_macro_beta,
    select_universe,
)
from .universe import load_universe

logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO").upper(),
    format="%(levelname)s — %(message)s",
    force=True,
)
LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# lazy fetch import (live only)
# --------------------------------------------------------------------------- #


def _import_fetch():
    from .fetch_data import (
        concat_data,
        get_fundamentals_fmp,
        get_news_sn,
        get_prices_polygon,
        score_sentiment_llm,
        get_dgs10_change,          # 🆕  10Y yield change
    )

    return (
        concat_data,
        get_fundamentals_fmp,
        get_news_sn,
        get_prices_polygon,
        score_sentiment_llm,
        get_dgs10_change,
    )


# --------------------------------------------------------------------------- #
# offline
# --------------------------------------------------------------------------- #


def _run_offline(roc_window: int, top_pct: int, out_dir: Path) -> None:
    df = load_universe()
    if "Close" not in df.columns:
        LOGGER.error("Universe CSV missing 'Close' column – offline abort")
        sys.exit(1)

    df["roc"] = calc_roc(df["Close"], window=roc_window)
    df["factor_score"] = calc_four_factor(df)
    selected = select_universe(df.fillna(0), top_pct=top_pct)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "offline_selected.csv"
    selected.to_csv(fp, index=False)
    print(f"✓ Offline → {fp}  ({len(selected)} rows)")


# --------------------------------------------------------------------------- #
# live
# --------------------------------------------------------------------------- #


def _run_live(  # noqa: PLR0913
    tickers: Sequence[str],
    start: date,
    end: date,
    roc_window: int,
    top_pct: int,
    out_dir: Path,
) -> None:
    (
        concat_data,
        get_fundamentals_fmp,
        get_news_sn,
        get_prices_polygon,
        score_sentiment_llm,
        get_dgs10_change,
    ) = _import_fetch()

    # 1) ár, fundamentum, hírszentiment
    closes = get_prices_polygon(tickers, start, end)
    fund = get_fundamentals_fmp(tickers)
    sent = pd.Series(
        {t: score_sentiment_llm(get_news_sn(t)) for t in tickers},
        name="Sentiment",
        dtype=float,
    )

    # 2) merge (outer)
    df_all = concat_data(closes, fund, sent)

    # 3) Fundamental + Technical + Macro
    df_all["Fundamental"] = calc_fundamental_composite(df_all)

    rsi_series = calc_technical_rsi(closes, window=20)
    df_all["Technical"] = df_all["symbol"].map(rsi_series).fillna(50.0)

    if "sector" not in df_all.columns:
        df_all["sector"] = "TECH"   # ideiglenes default
    dgs10_chg = get_dgs10_change()  # napi % változás
    df_all["Macro"] = calc_macro_beta(df_all, dgs10_chg)

    # 4) faktor-score
    df_all["factor_score"] = calc_four_factor(df_all)

    # 5) ROC – minden tickerre
    roc_series = closes.pct_change(roc_window).iloc[-1].mul(100)
    df_all["roc"] = df_all["symbol"].map(roc_series).fillna(0)

    # 6) NaN-robust adaptív momentum-szűrés
    df_all.fillna(0, inplace=True)
    if len(df_all) < max(10, 100 // max(1, top_pct)):
        selected = df_all.sort_values("factor_score", ascending=False)
    else:
        selected = select_universe(df_all, top_pct=top_pct)

    # 7) always write CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "live_selected.csv"
    selected.to_csv(fp, index=False)
    print(f"✓ Live → {fp}  ({len(selected)} rows)")

    # ── ÚJ: ROC-százalék JSON export ───────────────────────────────────
    (selected.set_index("symbol")["roc"]
             .round(2)                       # 2 tizedes
             .to_json(out_dir / "roc_strength.json", indent=2))
    # ───────────────────────────────────────────────────────────────────


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("portfolio-ai CLI")
    p.add_argument("mode", choices=["offline", "live"], nargs="?", default="offline")
    p.add_argument("--tickers", nargs="+")
    p.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--roc_window", type=int, default=21)
    p.add_argument("--top_pct", type=int, default=20)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / date.today().isoformat())
    return p.parse_args()


def _main() -> None:
    args = _parse()
    if args.mode == "offline":
        _run_offline(args.roc_window, args.top_pct, args.out)
        return

    if not args.tickers:
        LOGGER.error("live mode needs --tickers AAPL MSFT …")
        sys.exit(1)

    _run_live(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        roc_window=args.roc_window,
        top_pct=args.top_pct,
        out_dir=args.out,
    )


if __name__ == "__main__":
    _main()