import pandas as pd
from src.factors import calc_four_factor, calc_roc

def test_calc_four_factor_deterministic():
    df = pd.DataFrame({
        "Fundamental":[1,2], "Sentiment":[0.1,0.2],
        "Technical":[50,60], "Macro":[0.5,0.5]
    })
    out = calc_four_factor(df)
    # nagyobb Fund+Tech sor kapjon magasabb pontot
    assert out.iloc[1] > out.iloc[0]

def test_calc_roc_simple():
    s = pd.Series([100,110,121])
    roc = calc_roc(s, window=1)
    # (121-110)/110 ≈ 0.10 = 10 %
    assert abs(roc.iloc[-1] - 10) < 1e-6