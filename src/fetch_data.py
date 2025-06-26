"""fetch_data.py – unified data ingestion layer
Pulls prices, fundamentals, macro series and news-sentiment into pandas objects.
• Compatible with polygon-api-client ≥ 1.14  (timestamp/close attrs)
  and <1.14 (t/c attrs).
• Minimal exponential back-off.
.env keys: POLYGON_KEY  FMP_KEY  FRED_KEY  STOCKNEWS_KEY  OPENAI_API_KEY
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from functools import wraps
from typing import Iterable, Sequence

import pandas as pd
import requests
from fredapi import Fred
from openai import OpenAI
from polygon import RESTClient as PolygonClient

from .config import settings

LOGGER = logging.getLogger(__name__)
MAX_RETRY = 5


# ------------------------------------------------------------------ #
# retry decorator
# ------------------------------------------------------------------ #
def _retryable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sleep = 1.0
        for attempt in range(1, MAX_RETRY + 1):
            try:
                return func(*args, **kwargs)
            except (requests.HTTPError, requests.ConnectionError) as exc:
                if attempt == MAX_RETRY:
                    raise
                LOGGER.warning(
                    "%s failed (%s) – retry %d/%d", func.__name__, exc, attempt, MAX_RETRY
                )
                time.sleep(sleep)
                sleep *= 2

    return wrapper


# ------------------------------------------------------------------ #
# prices – Polygon
# ------------------------------------------------------------------ #
def _bar_to_point(bar) -> tuple[date, float]:
    """Return (date, close) tuple – works for old (t/c) and new (timestamp/close)."""
    ts = getattr(bar, "timestamp", getattr(bar, "t", None))
    close = getattr(bar, "close", getattr(bar, "c", None))
    if ts is None or close is None:
        raise AttributeError("Polygon Agg missing timestamp/close attributes")
    return datetime.fromtimestamp(ts / 1_000).date(), close


@_retryable
def get_prices_polygon(symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
    if settings.polygon_key is None:
        raise RuntimeError("POLYGON_KEY missing in .env")
    client = PolygonClient(api_key=settings.polygon_key)

    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        LOGGER.info("Polygon agg %s", sym)
        bars = client.get_aggs(sym, 1, "day", start.isoformat(), end.isoformat(), adjusted=True)
        closes[sym] = pd.Series(dict(_bar_to_point(b) for b in bars), name=sym)

    return pd.DataFrame(closes).sort_index()


# ------------------------------------------------------------------ #
# fundamentals – FMP
# ------------------------------------------------------------------ #
@_retryable
def get_fundamentals_fmp(symbols: Sequence[str]) -> pd.DataFrame:
    if settings.fmp_key is None:
        raise RuntimeError("FMP_KEY missing in .env")
    base = "https://financialmodelingprep.com/api/v3/key-metrics-ttm/{}?apikey={}".format

    rows: list[dict] = []
    for sym in symbols:
        r = requests.get(base(sym, settings.fmp_key), timeout=20)
        r.raise_for_status()
        data = r.json()
        if data:
            rows.append({**data[0], "symbol": sym})
        else:
            LOGGER.warning("FMP empty for %s", sym)

    return pd.DataFrame(rows).set_index("symbol")


# ------------------------------------------------------------------ #
# macro – FRED
# ------------------------------------------------------------------ #
fred_client: Fred | None = None


@_retryable
def get_macro_fred(series_ids: Sequence[str]) -> pd.DataFrame:
    global fred_client
    if fred_client is None:
        if settings.fred_key is None:
            raise RuntimeError("FRED_KEY missing in .env")
        fred_client = Fred(api_key=settings.fred_key)
    return pd.DataFrame({sid: fred_client.get_series_latest_release(sid) for sid in series_ids})

# --- FRED 10Y yield Δ% ------------------------------------------------------
@_retryable
def get_dgs10_change(days: int = 1) -> float:
    """
    Visszaadja a 10 éves US Treasury hozam (DGS10) *napi* %-os változását.
    Ha nincs friss adat (pl. hétvége), az utolsó két elérhető értékből számol.
    """
    series = get_macro_fred(["DGS10"])["DGS10"].dropna().astype(float).tail(2)
    if len(series) < 2:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-2] - 1) * 100)

# ------------------------------------------------------------------ #
# news + LLM sentiment
# ------------------------------------------------------------------ #
@_retryable
def get_news_sn(symbol: str, days: int = 7, items: int = 30) -> list[str]:
    if settings.stocknews_key is None:
        raise RuntimeError("STOCKNEWS_KEY missing")
    base = (
        "https://stocknewsapi.com/api/v1?tickers={}&items={}&token={}"
    ).format(symbol, items, settings.stocknews_key)

    # StockNews date param: last7days / last30days / … / MMDDYYYY
    if days <= 7:
        date_param = "last7days"
    elif days <= 30:
        date_param = "last30days"
    elif days <= 60:
        date_param = "last60days"
    elif days <= 90:
        date_param = "last90days"
    else:
        date_param = (date.today() - pd.Timedelta(days=days)).strftime("%m%d%Y")

    url = f"{base}&date={date_param}"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
    except requests.HTTPError as exc:
        LOGGER.warning("StockNews API error (%s) – empty list fallback", exc)
        data = []
    return [item["title"] for item in data]


openai_client: OpenAI | None = None


@_retryable
def score_sentiment_llm(texts: Iterable[str]) -> float:
    global openai_client
    if openai_client is None:
        if settings.openai_key is None:
            raise RuntimeError("OPENAI_API_KEY missing")
        openai_client = OpenAI(api_key=settings.openai_key)

    prompt = (
        "Return ONE float between -1 and 1 for overall sentiment.\n- "
        + "\n- ".join(texts)
    )
    chat = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    reply = chat.choices[0].message.content.strip()
    nums = re.findall(r"[-+]?[0-9]*\\.?[0-9]+", reply)
    if not nums and reply.replace(".", "", 1).lstrip("+-").isdigit():
        nums = [reply]
    return float(nums[-1]) if nums else 0.0


# ------------------------------------------------------------------ #
# merge helper
# ------------------------------------------------------------------ #
def concat_data(closes: pd.DataFrame, fundamentals: pd.DataFrame, sentiment: pd.Series) -> pd.DataFrame:
    latest_close = closes.iloc[-1].rename_axis("symbol").rename("Close")
    df = fundamentals.join(latest_close, how="outer")
    df = df.join(sentiment.rename("Sentiment"), how="outer")
    return df.reset_index()
