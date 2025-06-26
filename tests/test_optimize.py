import json, numpy as np, pandas as pd, importlib
from pathlib import Path
from hypothesis import given, strategies as st

# --- determinisztikus: sum≈1, nincs NaN ------------------------------------
def test_weights_sum_and_nan(tmp_path: Path):
    # dummy 3×3 kovariancia & mu
    sym = ["AAA", "BBB", "CCC"]
    Σ = np.eye(3)
    μ = pd.Series([0.1, 0.2, 0.15], index=sym)
    optimize = importlib.import_module("src.optimize")  # dinamikus import
    w, _ = optimize._solve_weights(μ, Σ)                # type: ignore
    assert abs(sum(w.values()) - 1) < 1e-6
    assert not any(np.isnan(list(w.values())))

# --- property: ROC>0 ⇒ súly∈[0,0.6] -----------------------------------------
@given(
    roc_vals=st.lists(st.floats(min_value=0.01, max_value=50), min_size=3, max_size=10)
)
def test_weights_with_positive_roc(roc_vals):
    n = len(roc_vals)
    sym = [f"T{i}" for i in range(n)]
    Σ = np.eye(n)
    μ = pd.Series(roc_vals, index=sym) / 100  # μ arányos ROC-cal
    optimize = importlib.import_module("src.optimize")  # dinamikus import
    w, _ = optimize._solve_weights(μ, Σ)                # type: ignore
    # minden súly 0–0.6 intervallumban
    assert all(0 <= v <= 0.6 + 1e-6 for v in w.values())