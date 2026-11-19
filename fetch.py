"""Fetch daily traditional-market food prices from PIHPS (Bank Indonesia) into data/raw/YYYY-MM.csv.

Backfill: python fetch.py --start 2017-01-01
Daily:    python fetch.py   (re-fetches the last 14 days to catch late or revised entries)

Safe to re-run: rows are upserted on (date, province, commodity_id).
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

URL = "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataKomoditas"
RAW = Path(__file__).parent / "data" / "raw"
KEY = ["date", "province", "commodity_id"]
COMMODITIES = {
    "com_3": "Beras Kualitas Medium I",  # stable control series
    "com_7": "Daging Ayam Ras Segar",
    "com_10": "Telur Ayam Ras Segar",
    "com_11": "Bawang Merah Ukuran Sedang",
    "com_12": "Bawang Putih Ukuran Sedang",
    "com_13": "Cabai Merah Besar",
    "com_14": "Cabai Merah Keriting",
    "com_15": "Cabai Rawit Hijau",
    "com_16": "Cabai Rawit Merah",
}
TRADITIONAL_MARKET = 1
DAILY_REPORT = 1
SLEEP_S = 5  # the server resets connections or fails the TLS handshake under bursts
ATTEMPTS = 6


def request(commodity_id: str, start: date, end: date) -> list[dict]:
    """One grid call: every province for one commodity, one column per day."""
    q = urllib.parse.urlencode({
        "price_type_id": TRADITIONAL_MARKET, "comcat_id": commodity_id, "province_id": "", "regency_id": "",
        "showKota": "false", "showPasar": "false", "tipe_laporan": DAILY_REPORT,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
    })
    req = urllib.request.Request(f"{URL}?{q}", headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["data"]
        except (OSError, ValueError) as e:  # connection reset, TLS failure, timeout, HTTP error, bad JSON
            if attempt == ATTEMPTS - 1:
                raise
            wait = 5 * 2**attempt  # 5 s .. 80 s
            print(f"  retry in {wait}s: {e}")
            time.sleep(wait)


def parse(rows: list[dict], commodity_id: str) -> list[dict]:
    """Wide grid rows (one per province, columns 'dd/mm/yyyy' -> '86,600') to long records. '-' means no data."""
    return [
        {
            "date": datetime.strptime(k, "%d/%m/%Y").date().isoformat(),
            "province": row["name"],
            "commodity_id": commodity_id,
            "commodity": COMMODITIES[commodity_id],
            "price": int(v.replace(",", "")),
        }
        for row in rows
        for k, v in row.items()
        if "/" in k and v not in ("-", "", None)
    ]


def save(records: list[dict]) -> None:
    """Upsert into one CSV per month, so daily runs only rewrite the current month's file."""
    df = pd.DataFrame(records)
    for month, part in df.groupby(df["date"].str[:7]):
        path = RAW / f"{month}.csv"
        if path.exists():
            part = pd.concat([pd.read_csv(path), part])
        part.drop_duplicates(KEY, keep="last").sort_values(KEY).to_csv(path, index=False)


def chunks(start: date, end: date, days: int = 90):
    """Split a date range: one request per ~quarter keeps responses small (half-year calls take ~10 s)."""
    while start <= end:
        stop = min(start + timedelta(days=days - 1), end)
        yield start, stop
        start = stop + timedelta(days=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date.today() - timedelta(days=14))
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--commodities", nargs="+", choices=COMMODITIES, default=list(COMMODITIES))
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    for cid in a.commodities:
        for s, e in chunks(a.start, a.end):
            try:
                records = parse(request(cid, s, e), cid)
            except (OSError, ValueError) as err:  # one bad chunk must not stop the other commodities
                failed.append(f"--commodities {cid} --start {s} --end {e}")
                print(f"{COMMODITIES[cid]:28} {s}..{e}: FAILED ({err})")
                continue
            print(f"{COMMODITIES[cid]:28} {s}..{e}: {len(records):>6} prices")
            if records:
                save(records)  # per chunk, so an interrupted backfill keeps its progress
            time.sleep(SLEEP_S)
    if failed:
        sys.exit("Failed chunks, re-run each with:\n" + "\n".join(f"  python fetch.py {f}" for f in failed))


if __name__ == "__main__":
    main()
