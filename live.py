"""Daily live run: release gate, 14-day forecasts, spike alerts, and scoring of past forecasts.

Run: python live.py
1. Gate: retrain today's recipe as if it were weeks ago and score it on the weeks since, which it never saw.
   Forecast check (last 4 weeks): it must not lose to the naive forecast and its P10-P90 range must hold
   70-90% of real prices. Warning check (last 13 weeks, since spikes are rare): the spike classifier must rank
   better than the "already rising" heuristic; with too few spikes to judge, the check is skipped, not failed.
   Result -> data/model_status.json.
2. Each part is published only if its own check passes: forecasts -> data/forecasts/YYYY-MM.csv, every calm
   series' spike probability -> data/alerts/YYYY-MM.csv. A failed part keeps its last published version and
   the job exits 1 (email).
Re-running for the same origin replaces that origin's rows.
"""
import json
import sys
from pathlib import Path

import pandas as pd

import backtest
import model
import quality
import spike
from model import HORIZON, stack

DATA = Path(__file__).parent / "data"
FORECASTS = DATA / "forecasts"
ALERTS = DATA / "alerts"
STATUS = DATA / "model_status.json"
SERIES = ["commodity_id", "province"]
COLUMNS = {"naive": "naive", "lightgbm_p10": "p10", "lightgbm": "p50", "lightgbm_p90": "p90"}
GATE_WEEKS = 4
SPIKE_GATE_WEEKS = 13  # 4 weeks of a calm market can hold spikes in only a handful of series: AP is then noise
MIN_SPIKE_SERIES = 20  # fewer distinct series with a spike than this: too little evidence to judge the warning
NAIVE_TOLERANCE = 1.02  # 4 weeks is a small sample; a 2% tie with naive is noise, not a broken model
COVERAGE_RANGE = (0.70, 0.90)


def verdict(s: dict) -> dict:
    """Pass/fail per published part, from the gate metrics."""
    forecast_ok = s["forecast_mape"] <= s["naive_mape"] * NAIVE_TOLERANCE and COVERAGE_RANGE[0] <= s["coverage"] <= COVERAGE_RANGE[1]
    judged = s["spike_series"] >= MIN_SPIKE_SERIES
    spike_ok = not judged or s["spike_ap"] > s["heuristic_ap"]
    return {"forecast_passed": bool(forecast_ok), "spike_judged": judged, "spike_passed": bool(spike_ok),
            "passed": bool(forecast_ok and spike_ok)}


def gate(df: pd.DataFrame, wide: pd.DataFrame) -> dict:
    end = wide.index[-1] - pd.Timedelta(days=HORIZON)  # last origin whose 14 days are over
    origins = pd.bdate_range(end - pd.Timedelta(weeks=GATE_WEEKS) + pd.Timedelta(days=1), end)
    spike_origins = pd.bdate_range(end - pd.Timedelta(weeks=SPIKE_GATE_WEEKS) + pd.Timedelta(days=1), end)
    never = 10**6  # train once, at the first holdout origin
    bt = backtest.run(df, [backtest.naive, model.LightGBM(refit_days=never)], origins).dropna()
    sp = spike.run(wide, test=spike_origins, refit_days=never)
    s = {
        "date": f"{wide.index[-1]:%Y-%m-%d}",
        "holdout": f"{origins[0]:%Y-%m-%d} .. {origins[-1]:%Y-%m-%d}",
        "forecast_mape": round(backtest.mape(bt["lightgbm"], bt["actual"]), 2),
        "naive_mape": round(backtest.mape(bt["naive"], bt["actual"]), 2),
        "coverage": round(float(bt["actual"].between(bt["lightgbm_p10"], bt["lightgbm_p90"]).mean()), 3),
        "spike_holdout": f"{spike_origins[0]:%Y-%m-%d} .. {spike_origins[-1]:%Y-%m-%d}",
        "spike_rate": round(float(sp["spike"].mean()), 3),
        "spike_series": int(sp[sp["spike"] == 1].groupby(SERIES, observed=True).ngroups),
        "spike_ap": round(spike.average_precision(sp["p"].to_numpy(), sp["spike"].to_numpy()), 3),
        "heuristic_ap": round(spike.average_precision(sp["to_level"].to_numpy(), sp["spike"].to_numpy()), 3),
    }
    return s | verdict(s)


def forecast(wide: pd.DataFrame) -> pd.DataFrame:
    origin = wide.index[-1]
    days = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON, name="date")
    preds = backtest.naive(wide) | model.LightGBM()(wide)
    out = pd.DataFrame({COLUMNS[name]: stack(pd.DataFrame(grid, index=days, columns=wide.columns)) for name, grid in preds.items()})
    out = out.dropna().round().astype(int).reset_index()  # stopped series have no forecast
    out = out[out["date"].dt.day_of_week < 5]  # PIHPS only reports on weekdays
    out.insert(0, "origin", origin)
    return out


def save(df: pd.DataFrame, folder: Path) -> Path:
    """Upsert one origin's rows into the month file."""
    origin = df["origin"].iloc[0]
    path = folder / f"{origin:%Y-%m}.csv"
    if path.exists():
        old = pd.read_csv(path, parse_dates=[c for c in ("origin", "date") if c in df])
        df = pd.concat([old[old["origin"] != origin], df])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values([c for c in ("origin", "date", *SERIES) if c in df]).to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def load(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    first = pd.read_csv(files[0], nrows=0).columns
    return pd.concat(pd.read_csv(p, parse_dates=[c for c in ("origin", "date") if c in first]) for p in files)


def score(prices: pd.DataFrame) -> pd.DataFrame:
    """Every saved forecast whose target day now has a reported price, with that price as `actual`."""
    f = load(FORECASTS)
    if f.empty:
        return f
    actual = prices[["date", *SERIES, "price"]].rename(columns={"price": "actual"})
    scored = f.merge(actual, on=["date", *SERIES])
    return scored.assign(h=(scored["date"] - scored["origin"]).dt.days)


def main() -> None:
    df = quality.clean(quality.load_raw())
    wide = backtest.to_wide(df)
    status = gate(df, wide)
    STATUS.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))
    failed = []
    if status["forecast_passed"]:
        f = forecast(wide)
        print(f"Origin {f['origin'].iloc[0]:%Y-%m-%d}: {len(f):,} forecasts -> {save(f, FORECASTS)}")
    else:
        failed.append("forecast")
    if status["spike_passed"]:
        a = spike.live(wide)
        print(f"{len(a)} calm series scored, {(a['p'] >= spike.ALERT_CUTOFF).sum()} alerts -> {save(a, ALERTS)}")
    else:
        failed.append("spike warning")
    if failed:
        sys.exit(f"Release gate failed for: {', '.join(failed)}. Keeping the last published version of those.")


if __name__ == "__main__":
    main()
