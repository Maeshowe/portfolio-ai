"""Factor utilities v2
====================
• Composite Fundamental: z(peRatio) + z(roe) → min-max scale 0-1
• Technical: 20-napos RSI (tisztán pandas, nincs TA-Lib)
• ROC, z-score, universe-filter változatlan; csak bővítjük az API-t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper – z-score, min-max
# ---------------------------------------------------------------------------

def _zscore(series: pd.Series) -> pd.Series:
    """NaN-robust z-score; ha σ=0 → minden elem 0."""
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _minmax01(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - s.min()) / rng

# ---------------------------------------------------------------------------
# 1. Composite Fundamental score
# ---------------------------------------------------------------------------

def calc_fundamental_composite(
    df: pd.DataFrame,
    pe_col: str = "peRatioTTM",
    roe_col: str = "roeTTM",
) -> pd.Series:
    """z(peRatio) + z(ROE) → 0-1 skála."""
    if pe_col not in df.columns or roe_col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, name="Fundamental")

    z_pe = _zscore(df[pe_col].replace(0, np.nan)) * -1  # alacsony PE = jobb
    z_roe = _zscore(df[roe_col])
    comp = _minmax01(z_pe.add(z_roe, fill_value=0))
    return comp.rename("Fundamental")

# ---------------------------------------------------------------------------
# 2. Technical – 20-napos RSI (0-100)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, window: int = 20) -> float:
    diff = series.diff().dropna()
    up = diff.clip(lower=0).rolling(window).mean()
    down = (-diff.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def calc_technical_rsi(closes: pd.DataFrame, window: int = 20) -> pd.Series:
    """Return last RSI per symbol as Series(index=symbol)."""
    return pd.Series({sym: _rsi(closes[sym], window) for sym in closes.columns}).rename("Technical")

# ---------------------------------------------------------------------------
# 2.5  Macro-β faktor  (szektorérzékenység × DGS10 Δ%)
# ---------------------------------------------------------------------------
SECTOR_BETA = {
    "TECH":  0.8,
    "CONS":  0.3,
    "IND":  -0.2,
    "UTIL": -0.6,
    "DEF":  -0.8,
}

def calc_macro_beta(df: pd.DataFrame, dgs10_change: float,
                    sector_col: str = "sector") -> pd.Series:
    """
    Beta × hozamváltozás → pozitív = kedvez, negatív = árt.
    Ha nincs szektor, 0-át ad.
    """
    beta = df[sector_col].str.upper().map(SECTOR_BETA).fillna(0)
    return (beta * dgs10_change).rename("Macro")

# ---------------------------------------------------------------------------
# 3. Four-factor composite (Fundamental, Sentiment, Technical, Macro)
# ---------------------------------------------------------------------------

def calc_four_factor(
    df: pd.DataFrame,
    cols: dict[str, str] | None = None,
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
) -> pd.Series:
    if cols is None:
        cols = {
            "fund": "Fundamental",
            "sent": "Sentiment",
            "tech": "Technical",
            "macro": "Macro",
        }
    if len(weights) != 4 or not abs(sum(weights) - 1) < 1e-6:
        raise ValueError("weights must sum to 1.0")

    z = {k: _zscore(df[c].fillna(0)) for k, c in cols.items()}
    comp = sum(w * z[k] for w, k in zip(weights, cols))
    return comp.rename("factor_score")

# ---------------------------------------------------------------------------
# 4. Momentum
# ---------------------------------------------------------------------------

def calc_roc(price: pd.Series, window: int = 21) -> pd.Series:  # noqa: D401
    if window <= 0:
        raise ValueError("window must be > 0")
    return (price / price.shift(window) - 1).mul(100).rename("roc")

# ---------------------------------------------------------------------------
# 5. Universe filter
# ---------------------------------------------------------------------------

def select_universe(df: pd.DataFrame, score_col: str = "factor_score", roc_col: str = "roc", top_pct: int = 20) -> pd.DataFrame:
    if not 0 < top_pct <= 100:
        raise ValueError("top_pct 1-100")

    df_valid = df.dropna(subset=[score_col, roc_col]).copy()
    if len(df_valid) < 10:
        return df_valid.sort_values(score_col, ascending=False).reset_index(drop=True)

    thresh = df_valid[roc_col].quantile(1 - top_pct / 100)
    return (
        df_valid[df_valid[roc_col] >= thresh]
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )
