import numpy as np
import pandas as pd

import model
import spike


def random_wide(days=500, seed=0):
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples([("com_16", "DKI Jakarta"), ("com_16", model.NATIONAL), ("com_3", "Bali")],
                                     names=["commodity_id", "province"])
    prices = 50_000 * np.exp(np.cumsum(rng.normal(0, 0.03, (days, len(cols))), axis=0))
    prices[rng.random(prices.shape) < 0.3] = np.nan  # missing reports, like weekends and holidays
    return pd.DataFrame(prices, index=pd.date_range("2024-01-01", periods=days, name="date"), columns=cols)


def test_features_at_a_date_do_not_depend_on_later_data():
    wide = random_wide()
    cut = wide.index[400]
    full = model.panels(np.log(wide.ffill(limit=10)))
    past = model.panels(np.log(wide.loc[:cut].ffill(limit=10)))
    for name in full:
        pd.testing.assert_series_equal(full[name].loc[cut], past[name].loc[cut], check_names=False)
    pd.testing.assert_series_equal(spike.levels(wide).loc[cut], spike.levels(wide.loc[:cut]).loc[cut], check_names=False)


def test_label_is_a_new_spike_within_the_horizon():
    days = pd.date_range("2026-01-01", periods=60, name="date")
    price = np.full(60, 100.0)
    price[40] = 160  # +60% jump for one day, above cabai rawit merah's 50% threshold
    wide = pd.DataFrame({("com_16", "DKI Jakarta"): price}, index=days)
    wide.columns.names = ["commodity_id", "province"]

    calm, label, _ = spike.labels(wide)
    s = label.iloc[:, 0]

    assert s.loc["2026-01-27":"2026-02-09"].eq(1).all()  # the 14 days before the jump see it coming
    assert s.loc["2026-01-26"] == 0  # 15 days before: outside the horizon
    assert not calm.iloc[:, 0].loc["2026-02-10"]  # on the jump itself the series is already waspada
