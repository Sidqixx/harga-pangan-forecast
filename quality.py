"""Data quality checks for data/raw/*.csv.

Run: python quality.py
Prints a report and exits 1 when the data is stale, so the daily job fails loudly.
Downstream code uses clean(load_raw()) to get rows that passed every check.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
SERIES = ["commodity_id", "province"]
PROVINCES = {
    "Semua Provinsi",  # national average
    "Aceh", "Bali", "Banten", "Bengkulu", "DI Yogyakarta", "DKI Jakarta", "Gorontalo", "Jambi", "Jawa Barat",
    "Jawa Tengah", "Jawa Timur", "Kalimantan Barat", "Kalimantan Selatan", "Kalimantan Tengah", "Kalimantan Timur",
    "Kalimantan Utara", "Kepulauan Bangka Belitung", "Kepulauan Riau", "Lampung", "Maluku", "Maluku Utara",
    "Nusa Tenggara Barat", "Nusa Tenggara Timur", "Papua", "Papua Barat", "Riau", "Sulawesi Barat",
    "Sulawesi Selatan", "Sulawesi Tengah", "Sulawesi Tenggara", "Sulawesi Utara", "Sumatera Barat",
    "Sumatera Selatan", "Sumatera Utara",
}
# A price is an outlier when it is outside this ratio to the median of the 21 surrounding observations of
# the same series. Typos (1,200 for bawang merah, 0 for beras) are one-off; real shocks such as cabai in
# Sumatera Utara in Dec 2025 persist for weeks, so the centered median moves with them and they pass.
OUTLIER_RATIO = (0.4, 2.5)
STALE_DAYS = 10  # Lebaran collective leave can pause reporting for about a week


def load_raw() -> pd.DataFrame:
    df = pd.concat(pd.read_csv(f) for f in sorted(RAW.glob("*.csv")))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(SERIES + ["date"], ignore_index=True)


def issues(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that fail a check, with a `rule` column. Expects load_raw() ordering."""
    unknown = ~df["province"].isin(PROVINCES)
    median = df.groupby(SERIES)["price"].transform(lambda s: s.rolling(21, center=True, min_periods=5).median())
    ratio = df["price"] / median
    outlier = ~unknown & ~ratio.between(*OUTLIER_RATIO)  # NaN ratio (series too short) also fails, on purpose
    return pd.concat([
        df[unknown].assign(rule="unknown_province"),
        df[outlier].assign(rule="outlier", ratio=ratio[outlier].round(2)),
    ])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(issues(df).index)


def main() -> None:
    df = load_raw()
    bad = issues(df)
    latest = df["date"].max()
    last_seen = df.drop(bad.index).groupby(SERIES)["date"].max()
    stopped = last_seen[last_seen < latest - pd.Timedelta(days=STALE_DAYS)]

    print(f"{len(df):,} rows, {df.groupby(SERIES).ngroups} series, {df['date'].min():%Y-%m-%d} .. {latest:%Y-%m-%d}")
    print(f"Dropped by clean(): {len(bad):,} rows ({bad['rule'].value_counts().to_dict()})")
    recent = bad[bad["date"] > latest - pd.Timedelta(days=30)]
    if len(recent):
        print("Issues in the last 30 days:")
        print(recent[["date", "province", "commodity", "price", "rule", "ratio"]].to_string(index=False))
    if len(stopped):
        print(f"Series with no data in the {STALE_DAYS} days before {latest:%Y-%m-%d}:")
        print(stopped.dt.strftime("%Y-%m-%d").to_string())

    age = (pd.Timestamp(date.today()) - latest).days
    if age > STALE_DAYS:
        sys.exit(f"STALE: newest price is {age} days old")


if __name__ == "__main__":
    main()
