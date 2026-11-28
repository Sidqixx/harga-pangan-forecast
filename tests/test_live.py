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


def test_gate_publishes_each_part_on_its_own_check():
    good_forecast = {"forecast_mape": 3.2, "naive_mape": 3.3, "coverage": 0.8}

    weak_warning = live.verdict(good_forecast | {"spike_series": 60, "spike_ap": 0.05, "heuristic_ap": 0.08})
    assert (weak_warning["forecast_passed"], weak_warning["spike_passed"], weak_warning["passed"]) == (True, False, False)

    # spikes in only 7 series: AP is noise, so the warning is not judged and not blocked
    too_few = live.verdict(good_forecast | {"spike_series": 7, "spike_ap": 0.05, "heuristic_ap": 0.08})
    assert (too_few["spike_judged"], too_few["spike_passed"], too_few["passed"]) == (False, True, True)
