import pandas as pd

import fetch


def test_parse_turns_wide_grid_into_long_rows_and_skips_missing():
    rows = [{"no": "II", "name": "DKI Jakarta", "level": 1, "02/11/2026": "65,850", "03/11/2026": "-"}]
    assert fetch.parse(rows, "com_16") == [
        {"date": "2026-11-02", "province": "DKI Jakarta", "commodity_id": "com_16",
         "commodity": "Cabai Rawit Merah", "price": 65850},
    ]


def test_save_is_idempotent_and_keeps_revised_price(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    row = {"date": "2026-11-02", "province": "DKI Jakarta", "commodity_id": "com_16", "commodity": "Cabai Rawit Merah"}
    fetch.save([{**row, "price": 65850}, {**row, "date": "2026-10-31", "price": 64000}])
    fetch.save([{**row, "price": 66000}])  # same day fetched again with a revised price

    nov = pd.read_csv(tmp_path / "2026-11.csv")
    assert nov["price"].tolist() == [66000]
    assert pd.read_csv(tmp_path / "2026-10.csv")["price"].tolist() == [64000]
