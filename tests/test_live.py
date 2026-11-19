import pandas as pd

import live


def test_rerun_for_same_origin_replaces_rows_and_score_joins_actuals(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "FORECASTS", tmp_path)
    row = {"origin": pd.Timestamp("2026-11-19"), "date": pd.Timestamp("2026-11-20"),
           "commodity_id": "com_16", "province": "DKI Jakarta", "naive": 76650, "p10": 75000, "p50": 76900, "p90": 80100}
    live.save(pd.DataFrame([row]), tmp_path)
    live.save(pd.DataFrame([{**row, "p50": 77000}]), tmp_path)  # same origin re-run

    prices = pd.DataFrame([{"date": pd.Timestamp("2026-11-20"), "commodity_id": "com_16", "province": "DKI Jakarta", "price": 78000}])
    scored = live.score(prices)

    assert scored[["p50", "actual", "h"]].values.tolist() == [[77000, 78000, 1]]
