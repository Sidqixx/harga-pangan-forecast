import pandas as pd

import lebaran


def test_distances_around_lebaran_2026():
    dates = pd.to_datetime(["2026-03-20", "2026-03-21", "2026-03-23", "2025-12-01"])

    until, since = lebaran.distances(pd.DatetimeIndex(dates))

    assert until.tolist() == [1, 0, 60, 60]  # 2027-03-09 is more than 60 days after 23 Mar 2026
    assert since.tolist() == [60, 60, 2, 60]  # on the day itself, `until` = 0 carries the signal
