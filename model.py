"""Global LightGBM quantile forecaster: one model per quantile, shared by every province x commodity series.

It predicts the log change from the last known price, so a Rp 13,000 rice series and a Rp 60,000 chili series
are on the same scale, then converts back to rupiah.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import lebaran
import quality

HORIZON = 14  # calendar days
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
RETURN_LAGS = (1, 3, 7, 14, 28, 56, 91, 364)
NATIONAL = "Semua Provinsi"
# shortcut: train only on Monday origins (about 1.4M rows instead of 7M) to keep a backtest refit under a minute;
# try every weekday if the model ends up starved for data.
TRAIN_WEEKDAY = 0
PARAMS = dict(
    objective="quantile", n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
    subsample=0.5, subsample_freq=1, colsample_bytree=0.8, seed=0, verbose=-1,
)


def stack(wide: pd.DataFrame) -> pd.Series:
    return wide.stack(["commodity_id", "province"], future_stack=True)


def panels(logp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Feature matrices (date x series), each value computed only from that date and earlier."""
    national = logp.xs(NATIONAL, axis=1, level="province").reindex(columns=logp.columns.get_level_values(0))
    national.columns = logp.columns  # every series gets its commodity's national price
    p = {f"ret_{k}": logp - logp.shift(k) for k in RETURN_LAGS}
    p["vol_28"] = logp.diff().rolling(28, min_periods=10).std()
    p["dev_28"] = logp - logp.rolling(28, min_periods=10).median()
    p["dev_91"] = logp - logp.rolling(91, min_periods=30).median()
    p["vs_national"] = logp - national
    p["national_ret_7"] = national - national.shift(7)
    p["national_ret_28"] = national - national.shift(28)
    return p


def design(logp: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (origin, series with a known price, horizon 1..HORIZON)."""
    p = panels(logp)
    base = pd.DataFrame({name: stack(f.loc[origins]) for name, f in p.items()})
    base["last"] = stack(logp.loc[origins])
    base = base.dropna(subset=["last"]).reset_index()
    parts = []
    for h in range(1, HORIZON + 1):
        target = base["date"] + pd.Timedelta(days=h)
        until, since = lebaran.distances(pd.DatetimeIndex(target))
        parts.append(base.assign(h=h, target=target, month=target.dt.month, lebaran_until=until, lebaran_since=since))
    x = pd.concat(parts, ignore_index=True)
    for level in ("commodity_id", "province"):  # fixed categories, so codes match between fit and predict
        x[level] = x[level].astype(pd.CategoricalDtype(sorted(logp.columns.unique(level))))
    return x


def feature_names(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if c not in ("date", "target", "last")]


class LightGBM:
    """Forecaster for backtest.run: called with the raw price history up to the origin, retrains when stale."""

    def __init__(self, refit_days: int = 182):
        # shortcut: half-yearly refits keep the 2-year backtest to ~5 fits; production retrains weekly (release 3)
        self.refit_days = refit_days
        self.models: dict[str, lgb.Booster] = {}
        self.trained_until = None

    def fit(self, history: pd.DataFrame) -> None:
        logp = np.log(history.ffill(limit=quality.STALE_DAYS))
        last_origin = history.index[-1] - pd.Timedelta(days=HORIZON)  # every training target must be known
        origins = logp.index[(logp.index.day_of_week == TRAIN_WEEKDAY) & (logp.index <= last_origin)]
        x = design(logp, origins)
        actual = stack(np.log(history))  # real reports only, not forward-filled days
        y = actual.reindex(pd.MultiIndex.from_arrays([x["target"], x["commodity_id"].astype(str), x["province"].astype(str)])).to_numpy() - x["last"].to_numpy()
        keep = ~np.isnan(y)
        features = feature_names(x)
        data = lgb.Dataset(x.loc[keep, features], y[keep], free_raw_data=False)
        self.models = {name: lgb.train({**PARAMS, "alpha": q}, data) for name, q in QUANTILES.items()}
        self.trained_until = history.index[-1]

    def __call__(self, history: pd.DataFrame) -> dict[str, np.ndarray]:
        origin = history.index[-1]
        if self.trained_until is None or origin - self.trained_until >= pd.Timedelta(days=self.refit_days):
            self.fit(history)
        logp = np.log(history.ffill(limit=quality.STALE_DAYS))
        x = design(logp, pd.DatetimeIndex([origin]))
        features = feature_names(x)
        cols = history.columns.get_indexer(pd.MultiIndex.from_arrays([x["commodity_id"].astype(str), x["province"].astype(str)]))
        preds = np.sort(np.column_stack([m.predict(x[features]) for m in self.models.values()]), axis=1)  # no crossing
        out = {}
        for i, name in enumerate(self.models):
            grid = np.full((HORIZON, history.shape[1]), np.nan)
            grid[x["h"].to_numpy() - 1, cols] = np.exp(x["last"].to_numpy() + preds[:, i])
            out[f"lightgbm_{name}" if name != "p50" else "lightgbm"] = grid
        return out
