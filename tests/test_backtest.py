import numpy as np
import pandas as pd

import backtest
from model import HORIZON


def test_seasonal_applies_last_years_move_over_the_same_days():
    days = pd.date_range("2025-01-01", periods=400, freq="D")
    history = pd.DataFrame({"s": np.arange(1.0, 401.0)}, index=days)  # price on day i = i + 1
    year_ago = history["s"].iloc[-1 - backtest.SEASON]  # price on the origin's day last year

    pred = backtest.seasonal(history)["seasonal"][:, 0]

    expected = 400.0 * (year_ago + np.arange(1, HORIZON + 1)) / year_ago
    np.testing.assert_allclose(pred, expected)
