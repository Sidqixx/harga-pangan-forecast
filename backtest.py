"""Rolling-origin backtest: how well does each forecaster predict the next 14 days?

Run: python backtest.py

Every Monday of the last two years is a forecast origin. Each forecaster only sees prices up to the origin,
predicts every calendar day 1..14 ahead, and is scored on the days that turned out to have a reported price.
"""
import numpy as np
import pandas as pd

import quality

HORIZON = 14  # calendar days
TEST_YEARS = 2
SEASON = 364  # 52 weeks, keeps the weekday aligned


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-daily price matrix, one column per (commodity_id, province); NaN on days without a report."""
    return df.pivot_table(index="date", columns=["commodity_id", "province"], values="price").asfreq("D")


def naive(history: pd.DataFrame) -> np.ndarray:
    """Every future day = the last known price."""
    return np.tile(history.iloc[-1].to_numpy(), (HORIZON, 1))


def seasonal(history: pd.DataFrame) -> np.ndarray:
    """Last known price, moved by how the price moved over the same 14 days one year earlier."""
    year_ago = history.iloc[-SEASON - 1:-SEASON + HORIZON].to_numpy()  # day origin-364 .. origin-350
    ratio = year_ago[1:] / year_ago[0]
    return np.where(np.isnan(ratio), 1, ratio) * history.iloc[-1].to_numpy()  # no last-year data: naive


FORECASTERS = {"naive": naive, "seasonal": seasonal}


def stack(wide: pd.DataFrame) -> pd.Series:
    return wide.stack(["commodity_id", "province"], future_stack=True)


def run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (origin, horizon, series) that has an actual price, with each forecaster's prediction."""
    wide = to_wide(df)
    known = wide.ffill(limit=quality.STALE_DAYS)  # carry a price forward at most 10 days; ffill only looks back
    end = wide.index.max() - pd.Timedelta(days=HORIZON)
    origins = pd.date_range(end - pd.DateOffset(years=TEST_YEARS), end, freq="W-MON")
    parts = []
    for origin in origins:
        actual = wide.loc[origin + pd.Timedelta(days=1):origin + pd.Timedelta(days=HORIZON)]
        history = known.loc[:origin]  # forecasters never see anything after the origin
        out = pd.DataFrame({"actual": stack(actual)})
        for name, forecaster in FORECASTERS.items():
            out[name] = stack(pd.DataFrame(forecaster(history), index=actual.index, columns=wide.columns))
        out = out.dropna().reset_index()
        out["origin"], out["h"] = origin, (out["date"] - origin).dt.days
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def mape(pred: pd.Series, actual: pd.Series) -> float:
    return float((pred - actual).abs().div(actual).mean() * 100)


def summary(bt: pd.DataFrame) -> pd.DataFrame:
    """MAPE (%) per commodity and horizon bucket for every forecaster."""
    bt = bt.assign(horizon=pd.cut(bt["h"], [0, 3, 7, HORIZON], labels=["1-3 d", "4-7 d", "8-14 d"]))
    names = quality.load_raw().drop_duplicates("commodity_id").set_index("commodity_id")["commodity"]
    rows = {
        (names[cid], hz): {f: mape(g[f], g["actual"]) for f in FORECASTERS}
        for (cid, hz), g in bt.groupby(["commodity_id", "horizon"], observed=True)
    }
    rows |= {("ALL", hz): {f: mape(g[f], g["actual"]) for f in FORECASTERS} for hz, g in bt.groupby("horizon", observed=True)}
    return pd.DataFrame(rows).T.round(1)


def main() -> None:
    bt = run(quality.clean(quality.load_raw()))
    print(f"{bt['origin'].nunique()} origins {bt['origin'].min():%Y-%m-%d} .. {bt['origin'].max():%Y-%m-%d}, {len(bt):,} scored predictions")
    print("MAPE (%), lower is better")
    print(summary(bt).to_string())


if __name__ == "__main__":
    main()
