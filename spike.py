"""Spike early warning: the probability that a series that is calm today jumps above its commodity's
alert level within the next 14 days.

Run: python spike.py   (backtest of the classifier against the "already rising" heuristic)

Alert level = (1 + threshold) x the median of the series' last 20 reports (about 4 weeks). Series already above
it are "waspada" today and shown as such; the warning is only about calm series, i.e. about new spikes.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import backtest
import lebaran
import model
import quality
from model import HORIZON, stack

# Chosen so that, in 2017-2024 data, a calm series turns into a spike within 14 days on about 10% of days:
# one rule for rice would fire on almost nothing, one rule for chili on almost everything. 5% floor for rice.
THRESHOLDS = {
    "com_3": 0.05,   # Beras Kualitas Medium I
    "com_7": 0.10,   # Daging Ayam Ras Segar
    "com_10": 0.10,  # Telur Ayam Ras Segar
    "com_11": 0.20,  # Bawang Merah Ukuran Sedang
    "com_12": 0.10,  # Bawang Putih Ukuran Sedang
    "com_13": 0.40,  # Cabai Merah Besar
    "com_14": 0.40,  # Cabai Merah Keriting
    "com_15": 0.30,  # Cabai Rawit Hijau
    "com_16": 0.50,  # Cabai Rawit Merah
}
ALERT_CUTOFF = 0.5  # backtest: ~9 alerts a day, 65% of them became a spike, 27% of all new spikes caught
MEDIAN_REPORTS = 20
PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
    subsample=0.5, subsample_freq=1, colsample_bytree=0.8, seed=0, verbose=-1,
)


def levels(wide: pd.DataFrame) -> pd.DataFrame:
    """Alert level per day and series, from the median of the last MEDIAN_REPORTS reported prices."""
    median = wide.apply(lambda s: s.dropna().rolling(MEDIAN_REPORTS, min_periods=15).median().reindex(s.index))
    theta = wide.columns.get_level_values("commodity_id").map(THRESHOLDS).to_numpy()
    return median.ffill(limit=quality.STALE_DAYS) * (1 + theta)


def labels(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(calm today, spike within HORIZON days or NaN if not knowable yet, today's price / alert level)."""
    level = levels(wide)
    price = wide.ffill(limit=quality.STALE_DAYS)
    future_max = wide[::-1].rolling(HORIZON, min_periods=1).max()[::-1].shift(-1)  # max report in (t, t+14]
    calm = price <= level  # False where unknown
    spike = (future_max > level).astype(float).where(calm & future_max.notna())
    spike.loc[wide.index > wide.index[-1] - pd.Timedelta(days=HORIZON)] = np.nan  # window not over yet
    return calm, spike, price / level


def design(wide: pd.DataFrame, origins: pd.DatetimeIndex, ratio: pd.DataFrame) -> pd.DataFrame:
    """One row per (origin, series) with the forecaster's features plus closeness to the alert level."""
    logp = np.log(wide.ffill(limit=quality.STALE_DAYS))
    frames = model.panels(logp) | {"to_level": ratio}
    x = pd.DataFrame({name: stack(f.loc[origins]) for name, f in frames.items()})
    dates = pd.DatetimeIndex(x.index.get_level_values("date"))
    until, since = lebaran.distances(dates)
    x = x.assign(month=dates.month, lebaran_until=until, lebaran_since=since).reset_index()
    for level in ("commodity_id", "province"):
        x[level] = x[level].astype(pd.CategoricalDtype(sorted(wide.columns.unique(level))))
    return x


def features(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if c != "date"]


def fit(x: pd.DataFrame, y: np.ndarray) -> lgb.Booster:
    return lgb.train(PARAMS, lgb.Dataset(x[features(x)], y))


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve: mean precision at the rank of every true spike."""
    y = y[np.argsort(-score, kind="stable")]
    hits = np.cumsum(y)
    return float((hits / np.arange(1, len(y) + 1))[y == 1].mean())


def run(wide: pd.DataFrame, test: pd.DatetimeIndex | None = None, refit_days: int = 182) -> pd.DataFrame:
    """Score each calm series on every test weekday (default: last 2 years), then compare with what happened.

    Features use only trailing windows (tests/test_spike.py checks this), so they are computed once on the full
    data; each refit trains only on origins whose 14-day outcome was known before the refit date.
    """
    calm, spike, ratio = labels(wide)
    if test is None:
        end = wide.index[-1] - pd.Timedelta(days=HORIZON)
        test = pd.bdate_range(end - pd.DateOffset(years=backtest.TEST_YEARS), end)
    train_days = wide.index[(wide.index.day_of_week < 5) & (wide.index < test[0])]
    x_all = design(wide, train_days.append(test), ratio)
    y_all = stack(spike.loc[train_days.append(test)]).to_numpy()
    parts = []
    for start in pd.date_range(test[0], test[-1], freq=f"{refit_days}D"):
        known_until = start - pd.Timedelta(days=HORIZON)  # outcomes visible on the refit date
        train = (x_all["date"] <= known_until) & ~np.isnan(y_all)
        booster = fit(x_all[train], y_all[train])
        block = (x_all["date"] >= start) & (x_all["date"] < start + pd.Timedelta(days=refit_days)) & ~np.isnan(y_all)
        parts.append(x_all[block][["date", "commodity_id", "province", "to_level"]].assign(
            p=booster.predict(x_all.loc[block, features(x_all)]), spike=y_all[block]))
    return pd.concat(parts, ignore_index=True)


def live(wide: pd.DataFrame) -> pd.DataFrame:
    """Train on every weekday whose 14-day outcome is known, then score today's calm series."""
    calm, spike, ratio = labels(wide)
    days = wide.index[wide.index.day_of_week < 5]
    x = design(wide, days, ratio)
    y = stack(spike.loc[days]).to_numpy()
    booster = fit(x[~np.isnan(y)], y[~np.isnan(y)])
    today = wide.index[-1:]
    now = design(wide, today, ratio).assign(  # same row order as stack() of today's row
        price=stack(wide.ffill(limit=quality.STALE_DAYS).loc[today]).to_numpy().round(),
        level=stack(levels(wide).loc[today]).to_numpy().round(),
    )[stack(calm.loc[today]).to_numpy()].astype({"price": int, "level": int})
    now["p"] = booster.predict(now[features(x)]).round(3)
    return now[["date", "commodity_id", "province", "price", "level", "p"]].rename(columns={"date": "origin"})


def score_alerts(alerts: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Past alerts whose 14 days are over, with `spike` = 1 if the series really crossed its alert level."""
    _, spike, _ = labels(wide)
    outcome = stack(spike).dropna().rename("spike").reset_index().rename(columns={"date": "origin"})
    return alerts.merge(outcome, on=["origin", "commodity_id", "province"])


def summary(bt: pd.DataFrame) -> None:
    rate = bt["spike"].mean()
    print(f"{bt['date'].nunique()} origins, {len(bt):,} calm series-days, new-spike rate {rate:.1%}")
    print("Average precision (higher is better; random = the spike rate):")
    print(f"  classifier                  {average_precision(bt['p'].to_numpy(), bt['spike'].to_numpy()):.3f}")
    print(f"  heuristic (closest to level) {average_precision(bt['to_level'].to_numpy(), bt['spike'].to_numpy()):.3f}")
    top = bt.sort_values("p", ascending=False).groupby("date").head(10)
    print(f"Top 10 alerts per day: {top['spike'].mean():.1%} turned into a spike")
    print("Alert cut-off  alerts/day  precision  recall")
    for cut in (0.3, 0.4, 0.5, 0.6, 0.7):
        a = bt[bt["p"] >= cut]
        print(f"  p >= {cut:.1f}     {len(a) / bt['date'].nunique():9.1f}  {a['spike'].mean():9.1%}  {a['spike'].sum() / bt['spike'].sum():6.1%}")
    names = quality.load_raw().drop_duplicates("commodity_id").set_index("commodity_id")["commodity"]
    print("Per commodity: spike rate, AP classifier, AP heuristic")
    for cid, g in bt.groupby("commodity_id", observed=True):
        print(f"  {names[cid]:28} {g['spike'].mean():6.1%}  {average_precision(g['p'].to_numpy(), g['spike'].to_numpy()):.3f}"
              f"  {average_precision(g['to_level'].to_numpy(), g['spike'].to_numpy()):.3f}")


if __name__ == "__main__":
    summary(run(backtest.to_wide(quality.clean(quality.load_raw()))))
