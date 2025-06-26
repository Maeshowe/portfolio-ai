"""
Globális konfiguráció – útvonalak és környezeti változók betöltése.
PEP 8-kompatibilis, inline kommentekkel.
"""
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
import os

# Betöltjük a .env-et (ha létezik a projekt gyökerében)
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)  # override=False → shell > .env

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)  # biztos, hogy létezik

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

@dataclass(frozen=True)
class Settings:
    fmp_key: str | None = os.getenv("FMP_KEY")
    polygon_key: str | None = os.getenv("POLYGON_KEY")
    fred_key: str | None = os.getenv("FRED_KEY")
    stocknews_key: str | None = os.getenv("STOCKNEWS_KEY")
    openai_key: str | None = os.getenv("OPENAI_API_KEY")

settings = Settings()  # importtal egyből elérhető