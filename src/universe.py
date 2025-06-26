"""
Ticker-universum kezelés: manuális CSV-hozzáadás és betöltés.
Futtatható modulként is (python -m src.universe <csv_path>).
"""
from pathlib import Path
import shutil
import argparse
import pandas as pd
from .config import DATA_DIR

# --- állandó utak ----------------------------------------------------------- #
UNIVERSE_DIR = DATA_DIR / "universe"
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
LATEST = UNIVERSE_DIR / "latest.csv"          # symlink vagy fájl

# --- publikus függvények ---------------------------------------------------- #
def load_universe() -> pd.DataFrame:
    """Betölti az aktuális *latest.csv*-t DataFrame-be."""
    if not LATEST.exists():
        raise FileNotFoundError(
            "Universe file not found. "
            "Add one with: python -m src.universe /path/to/TU.csv"
        )
    return pd.read_csv(LATEST)

def add_universe_csv(csv_path: str | Path) -> None:
    """
    Új TU_YYYY-MM-DD.csv bemásolása és a *latest.csv* symlink frissítése.
    """
    src = Path(csv_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    dest = UNIVERSE_DIR / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    else:
        print("↻ Fájl már a célmappában – másolás kihagyva")
    if LATEST.exists() or LATEST.is_symlink():
        LATEST.unlink()
    # relatív cél, mert ugyanabban a mappában van
    LATEST.symlink_to(dest.name)
    print(f"✓ Universe frissítve → {dest.name}")

# --- CLI (python -m src.universe <csv>) ------------------------------------- #
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Add new ticker-universe CSV.")
    ap.add_argument("csv_path", help="Path to TU_YYYY-MM-DD.csv")
    args = ap.parse_args()
    add_universe_csv(args.csv_path)

if __name__ == "__main__":
    _cli()