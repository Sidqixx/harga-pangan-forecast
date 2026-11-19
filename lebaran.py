"""Idul Fitri dates and the distance features the model uses."""
import numpy as np
import pandas as pd

# 1 Syawal as set by the government's sidang isbat. 2027 is Muhammadiyah's date; the government decides in
# March 2027 and may differ by a day, so update it then.
LEBARAN = pd.to_datetime([
    "2017-06-25", "2018-06-15", "2019-06-05", "2020-05-24", "2021-05-13",
    "2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31", "2026-03-21", "2027-03-09",
])
WINDOW = 60  # days; further away the holiday has no visible effect on prices, so distances are capped


def distances(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(days until the next Lebaran, days since the last one), both capped at WINDOW."""
    d = dates.values.astype("datetime64[D]")
    leb = LEBARAN.values.astype("datetime64[D]")
    i = np.searchsorted(leb, d)  # first Lebaran on or after each date
    until = np.where(i < len(leb), (leb[np.minimum(i, len(leb) - 1)] - d).astype(float), WINDOW)
    since = np.where(i > 0, (d - leb[np.maximum(i - 1, 0)]).astype(float), WINDOW)
    return np.minimum(until, WINDOW), np.minimum(since, WINDOW)
