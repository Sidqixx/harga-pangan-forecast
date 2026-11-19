import pandas as pd

import quality


def series(prices, province="DKI Jakarta"):
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=len(prices)),
        "province": province, "commodity_id": "com_16", "commodity": "Cabai Rawit Merah", "price": prices,
    })


def test_flags_one_off_typo_but_keeps_a_sustained_price_shock():
    typo = [50_000] * 15 + [1_200] + [50_000] * 15
    shock = [50_000] * 15 + [150_000] * 15  # tripled and stayed there: a real event, not an input error
    df = pd.concat([series(typo), series(shock, province="Jawa Barat")], ignore_index=True)

    bad = quality.issues(df)

    assert bad[["province", "price", "rule"]].values.tolist() == [["DKI Jakarta", 1_200, "outlier"]]


def test_flags_unknown_province_and_clean_drops_it():
    df = pd.concat([series([50_000] * 10), series([50_000] * 10, province="Dum")], ignore_index=True)

    assert set(quality.issues(df)["rule"]) == {"unknown_province"}
    assert set(quality.clean(df)["province"]) == {"DKI Jakarta"}
