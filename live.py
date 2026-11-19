"""Daily live forecast, and scoring of every past forecast once the real price is out.

Run: python live.py
Trains on all data up to the newest price (the origin), then saves the next 14 days for every active series
to data/forecasts/YYYY-MM.csv. Re-running for the same origin replaces that origin's rows.
"""
from pathlib import Path

import pandas as pd

import backtest
import model
import quality
from model import HORIZON, stack

FORECASTS = Path(__file__).parent / "data" / "forecasts"
SERIES = ["commodity_id", "province"]
COLUMNS = {"naive": "naive", "lightgbm_p10": "p10", "lightgbm": "p50", "lightgbm_p90": "p90"}


def forecast(wide: pd.DataFrame) -> pd.DataFrame:
    origin = wide.index[-1]
    days = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON, name="date")
    preds = backtest.naive(wide) | model.LightGBM()(wide)
    out = pd.DataFrame({COLUMNS[name]: stack(pd.DataFrame(grid, index=days, columns=wide.columns)) for name, grid in preds.items()})
    out = out.dropna().round().astype(int).reset_index()  # stopped series have no forecast
    out = out[out["date"].dt.day_of_week < 5]  # PIHPS only reports on weekdays
    out.insert(0, "origin", origin)
    return out


def save(df: pd.DataFrame) -> Path:
    origin = df["origin"].iloc[0]
    path = FORECASTS / f"{origin:%Y-%m}.csv"
    if path.exists():
        old = pd.read_csv(path, parse_dates=["origin", "date"])
        df = pd.concat([old[old["origin"] != origin], df])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["origin", "date", *SERIES]).to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def score(prices: pd.DataFrame) -> pd.DataFrame:
    """Every saved forecast whose target day now has a reported price, with that price as `actual`."""
    files = sorted(FORECASTS.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    f = pd.concat(pd.read_csv(p, parse_dates=["origin", "date"]) for p in files)
    actual = prices[["date", *SERIES, "price"]].rename(columns={"price": "actual"})
    scored = f.merge(actual, on=["date", *SERIES])
    return scored.assign(h=(scored["date"] - scored["origin"]).dt.days)


def main() -> None:
    wide = backtest.to_wide(quality.clean(quality.load_raw()))
    df = forecast(wide)
    print(f"Origin {df['origin'].iloc[0]:%Y-%m-%d}: {len(df):,} forecasts for {df.groupby(SERIES).ngroups} series -> {save(df)}")


if __name__ == "__main__":
    main()
