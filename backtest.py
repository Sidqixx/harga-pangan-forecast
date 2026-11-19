"""Rolling-origin backtest: how well does each forecaster predict the next 14 days?

Run: python backtest.py

Every Monday of the last two years is a forecast origin. Each forecaster only sees prices up to the origin,
predicts every calendar day 1..14 ahead, and is scored on the days that turned out to have a reported price.
"""
import numpy as np
import pandas as pd

import model
import quality
from model import HORIZON, stack

TEST_YEARS = 2
SEASON = 364  # 52 weeks, keeps the weekday aligned


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-daily price matrix, one column per (commodity_id, province); NaN on days without a report."""
    return df.pivot_table(index="date", columns=["commodity_id", "province"], values="price").asfreq("D")


def known(history: pd.DataFrame) -> pd.DataFrame:
    """Carry each price forward at most STALE_DAYS; after that the series counts as stopped."""
    return history.ffill(limit=quality.STALE_DAYS)


def naive(history: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every future day = the last known price."""
    return {"naive": np.tile(known(history).iloc[-1].to_numpy(), (HORIZON, 1))}


def seasonal(history: pd.DataFrame) -> dict[str, np.ndarray]:
    """Last known price, moved by how the price moved over the same 14 days one year earlier."""
    k = known(history)
    year_ago = k.iloc[-SEASON - 1:-SEASON + HORIZON].to_numpy()  # day origin-364 .. origin-350
    ratio = year_ago[1:] / year_ago[0]
    return {"seasonal": np.where(np.isnan(ratio), 1, ratio) * k.iloc[-1].to_numpy()}  # no last-year data: naive


def run(df: pd.DataFrame, forecasters: list) -> pd.DataFrame:
    """One row per (origin, horizon, series) that has an actual price, with every forecaster's prediction."""
    wide = to_wide(df)
    end = wide.index.max() - pd.Timedelta(days=HORIZON)
    origins = pd.date_range(end - pd.DateOffset(years=TEST_YEARS), end, freq="W-MON")
    parts = []
    for origin in origins:
        history = wide.loc[:origin]  # forecasters never see anything after the origin
        actual = wide.loc[origin + pd.Timedelta(days=1):origin + pd.Timedelta(days=HORIZON)]
        out = pd.DataFrame({"actual": stack(actual)})
        for forecaster in forecasters:
            for name, grid in forecaster(history).items():
                out[name] = stack(pd.DataFrame(grid, index=actual.index, columns=wide.columns))
        out = out.dropna(subset=["actual"]).reset_index()
        out["origin"], out["h"] = origin, (out["date"] - origin).dt.days
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def mape(pred: pd.Series, actual: pd.Series) -> float:
    return float((pred - actual).abs().div(actual).mean() * 100)


def summary(bt: pd.DataFrame) -> pd.DataFrame:
    """MAPE (%) per commodity and horizon bucket, plus how often the actual fell inside LightGBM's P10-P90."""
    bt = bt.dropna().assign(horizon=pd.cut(bt["h"], [0, 3, 7, HORIZON], labels=["1-3 d", "4-7 d", "8-14 d"]))
    points = [c for c in bt.columns if c in ("naive", "seasonal", "lightgbm")]
    names = quality.load_raw().drop_duplicates("commodity_id").set_index("commodity_id")["commodity"]

    def score(g: pd.DataFrame) -> dict:
        s = {f: mape(g[f], g["actual"]) for f in points}
        if "lightgbm_p10" in g:
            s["p10-p90 coverage"] = float(g["actual"].between(g["lightgbm_p10"], g["lightgbm_p90"]).mean() * 100)
        return s

    rows = {(names[cid], hz): score(g) for (cid, hz), g in bt.groupby(["commodity_id", "horizon"], observed=True)}
    rows |= {("ALL", hz): score(g) for hz, g in bt.groupby("horizon", observed=True)}
    return pd.DataFrame(rows).T.sort_index().round(1)


def main() -> None:
    bt = run(quality.clean(quality.load_raw()), [naive, seasonal, model.LightGBM()])
    print(f"{bt['origin'].nunique()} origins {bt['origin'].min():%Y-%m-%d} .. {bt['origin'].max():%Y-%m-%d}, {len(bt):,} scored predictions")
    print("MAPE (%), lower is better; coverage should be close to 80")
    print(summary(bt).to_string())


if __name__ == "__main__":
    main()
