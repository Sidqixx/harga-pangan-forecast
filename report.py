"""Build the public dashboard (static; rebuilt by the daily job, served by GitHub Pages).

Run: python report.py
Writes site/index.html (page + today's summary inline) and site/series/<commodity_id>.json, one file per
commodity that the page loads only when the user picks it, so the first load stays small.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import backtest
import lebaran
import live
import quality
import spike
from model import HORIZON

ROOT = Path(__file__).parent
SITE = ROOT / "site"
TEMPLATE = ROOT / "web" / "index.html"
NATIONAL = "Semua Provinsi"
SHORT = {  # chip labels; full names are too long for a phone row
    "com_3": "Beras", "com_7": "Ayam", "com_10": "Telur", "com_11": "Bawang merah", "com_12": "Bawang putih",
    "com_13": "Cabai merah besar", "com_14": "Cabai keriting", "com_15": "Cabai rawit hijau", "com_16": "Cabai rawit merah",
}


def ints(values) -> list:
    """JSON has no NaN: missing prices become null, prices become whole rupiah."""
    return [None if pd.isna(v) else int(round(v)) for v in values]


def names(df: pd.DataFrame) -> pd.Series:
    return df.drop_duplicates("commodity_id").set_index("commodity_id")["commodity"]


def status(df: pd.DataFrame, wide: pd.DataFrame, level: pd.Series) -> pd.DataFrame:
    """Latest price of every active series vs its 4-week median and its commodity's alert level."""
    cid = wide.columns.get_level_values("commodity_id")
    theta = pd.Series(cid.map(spike.THRESHOLDS), index=wide.columns)
    price = wide.ffill(limit=quality.STALE_DAYS).iloc[-1]
    median = level / (1 + theta)
    return pd.DataFrame({
        "cid": cid, "province": wide.columns.get_level_values("province"),
        "price": price.to_numpy(), "change": (price / median - 1).to_numpy(), "threshold": theta.to_numpy(),
    }).dropna()  # stopped or too-short series


def lebaran_curves(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """National price around each Lebaran vs its level 45-60 days before, median over the years (%)."""
    daily = df[df["province"] == NATIONAL].pivot(index="date", columns="commodity_id", values="price").asfreq("D").ffill()
    offsets = range(-30, 31)
    curves = []
    for day in lebaran.LEBARAN[lebaran.LEBARAN <= daily.index.max() - pd.Timedelta(days=30)]:
        base = daily.loc[day - pd.Timedelta(days=60):day - pd.Timedelta(days=45)].median()
        curves.append((daily.reindex([day + pd.Timedelta(days=o) for o in offsets]) / base - 1).set_axis(offsets) * 100)
    return pd.concat(curves).groupby(level=0).median(), len(curves)


def latest(folder: Path) -> pd.DataFrame:
    rows = live.load(folder)
    return rows[rows["origin"] == rows["origin"].max()] if len(rows) else rows


def series_files(df: pd.DataFrame, wide: pd.DataFrame, level: pd.Series) -> None:
    """Per commodity: last year daily, full history weekly, alert levels, today's forecast, Lebaran curve."""
    recent = wide.loc[wide.index[-1] - pd.Timedelta(days=365):]
    recent = recent[recent.notna().any(axis=1)]  # report days only
    weekly = wide.resample("W").median()
    forecast = latest(live.FORECASTS)
    alerts = latest(live.ALERTS)
    curves, years = lebaran_curves(df)
    out = SITE / "series"
    out.mkdir(parents=True, exist_ok=True)
    for cid in wide.columns.unique("commodity_id"):
        provinces = wide[cid].columns
        f = forecast[forecast["commodity_id"] == cid]
        a = alerts[alerts["commodity_id"] == cid] if len(alerts) else alerts
        days = sorted(f["date"].unique())
        grid = {q: f.pivot(index="date", columns="province", values=q).reindex(days) for q in ("p10", "p50", "p90")}
        data = {
            "dates": [f"{d:%Y-%m-%d}" for d in recent.index],
            "prices": {p: ints(recent[(cid, p)]) for p in provinces},
            "weekly": {"dates": [f"{d:%Y-%m-%d}" for d in weekly.index], "prices": {p: ints(weekly[(cid, p)]) for p in provinces}},
            "level": {p: v for p, v in zip(provinces, ints(level[cid])) if v is not None},
            "forecast": {"dates": [f"{d:%Y-%m-%d}" for d in days],
                         **{q: {p: ints(g[p]) for p in g.columns} for q, g in grid.items()}},
            "lebaran": {"offsets": list(curves.index), "values": [round(v, 1) for v in curves[cid].fillna(0)], "years": years},
            "p": dict(zip(a["province"], a["p"])) if len(a) else {},  # spike probability of every calm series today
        }
        (out / f"{cid}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def summary(df: pd.DataFrame, wide: pd.DataFrame, level: pd.Series) -> dict:
    st = status(df, wide, level)
    waspada = st[st["change"] > st["threshold"]].sort_values("change", ascending=False)
    alerts = latest(live.ALERTS)
    hot = alerts[alerts["p"] >= spike.ALERT_CUTOFF].sort_values("p", ascending=False) if len(alerts) else alerts
    n = names(df)
    notices = []
    age = (pd.Timestamp(datetime.now(timezone.utc).date()) - df["date"].max()).days
    if age > quality.STALE_DAYS:
        notices.append(f"Harga terbaru dari PIHPS sudah {age} hari. Angka di halaman ini mungkin belum mencerminkan pasar saat ini.")
    gate = gate_status()
    if gate and not gate["forecast_passed"]:
        notices.append("Pemeriksaan model perkiraan terakhir tidak lolos. Perkiraan di bawah adalah versi terakhir yang lolos.")
    if gate and not gate["spike_passed"]:
        notices.append("Pemeriksaan model peringatan terakhir tidak lolos. Daftar peringatan di bawah adalah versi terakhir yang lolos.")
    return {
        "dataDate": tanggal(df["date"].max()),
        "built": f"{tanggal(datetime.now(timezone.utc))} {datetime.now(timezone.utc):%H:%M} UTC",
        "national": NATIONAL,
        "notices": notices,
        "commodities": [{"id": c, "name": n[c], "short": SHORT[c], "threshold": spike.THRESHOLDS[c]} for c in sorted(n.index, key=SHORT.get)],
        "series": len(st),
        "waspada": [{"cid": r.cid, "province": r.province, "price": int(r.price), "change": round(r.change, 3),
                     "threshold": r.threshold} for r in waspada.itertuples()],
        "alerts": [{"cid": r.commodity_id, "province": r.province, "price": int(r.price), "level": int(r.level), "p": r.p}
                   for r in hot.itertuples()],
        "alertCutoff": spike.ALERT_CUTOFF,
    }


BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def tanggal(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {BULAN[ts.month - 1]} {ts.year}"


def angka(v: float, digits: int = 1) -> str:
    """Indonesian number format: 1.234 and 3,2."""
    return f"{v:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def alert_track(df: pd.DataFrame, wide: pd.DataFrame) -> str:
    alerts = live.load(live.ALERTS)
    past = spike.score_alerts(alerts[alerts["p"] >= spike.ALERT_CUTOFF], wide) if len(alerts) else alerts
    if not len(past):
        return "Rekam jejak live mulai tampil setelah 14 hari."
    return (f"Live: <b>{angka(past['spike'].mean() * 100, 0)}%</b> dari {angka(len(past), 0)} peringatan "
            f"yang sudah lewat 14 hari memang melonjak.")


def live_table(df: pd.DataFrame) -> str:
    """Accuracy of the daily forecasts so far, scored against prices that have since been reported."""
    s = live.score(df)
    if s.empty:
        return "<p>Belum ada perkiraan yang bisa dinilai karena harga aslinya belum keluar.</p>"
    s = s.assign(horizon=pd.cut(s["h"], [0, 3, 7, HORIZON], labels=["1 sampai 3 hari", "4 sampai 7 hari", "8 sampai 14 hari"]))
    table = pd.DataFrame([
        {
            "Perkiraan untuk": hz, "Sudah dinilai": len(g),
            "Meleset, tebakan harga terakhir": backtest.mape(g["naive"], g["actual"]),
            "Meleset, model": backtest.mape(g["p50"], g["actual"]),
            "Harga asli di rentang 80%": g["actual"].between(g["p10"], g["p90"]).mean() * 100,
        }
        for hz, g in s.groupby("horizon", observed=True)
    ]).to_html(index=False, border=0, formatters={
        "Sudah dinilai": lambda v: angka(v, 0), "Meleset, tebakan harga terakhir": lambda v: angka(v) + "%",
        "Meleset, model": lambda v: angka(v) + "%", "Harga asli di rentang 80%": lambda v: angka(v, 0) + "%",
    })
    return (f"<p>Setiap perkiraan dicocokkan dengan harga asli begitu harganya keluar. Perkiraan harian berjalan sejak "
            f"<b>{tanggal(s['origin'].min())}</b>, dan {angka(len(s), 0)} sudah dinilai. Meleset adalah rata-rata selisih "
            f"dalam persen, makin kecil makin baik. Kolom terakhir idealnya sekitar 80%.</p>\n"
            f'<div class="table-wrap" tabindex="0" role="region" aria-label="Tabel akurasi perkiraan">{table}</div>')


def gate_status() -> dict | None:
    if not live.STATUS.exists():
        return None
    s = json.loads(live.STATUS.read_text())
    s.setdefault("forecast_passed", s["passed"])  # files written before the gate was split
    s.setdefault("spike_passed", s["passed"])
    s.setdefault("spike_judged", True)
    s.setdefault("spike_holdout", s["holdout"])
    return s


def gate_line() -> str:
    s = gate_status()
    if s is None:
        return ""
    start, end = s["holdout"].split(" .. ")
    sp_start, sp_end = s["spike_holdout"].split(" .. ")
    ok = lambda passed: "<b>lolos</b>" if passed else "<b>tidak lolos</b>, yang tampil versi terakhir yang lolos"
    if not s["spike_judged"]:
        spike_text = (f"Peringatan: belum dinilai, karena dalam {tanggal(sp_start)} sampai {tanggal(sp_end)} "
                      f"lonjakan hanya terjadi di {s['spike_series']} harga, terlalu sedikit untuk menilai.")
    else:
        spike_text = (f"Peringatan: {ok(s['spike_passed'])}. Diuji pada {tanggal(sp_start)} sampai {tanggal(sp_end)}, "
                      f"skor ketepatannya {angka(s['spike_ap'], 2)} (aturan sederhana: {angka(s['heuristic_ap'], 2)}; skala 0 sampai 1).")
    return (f"Pemeriksaan model {tanggal(s['date'])}, pada periode yang belum pernah dilihat model. "
            f"Perkiraan: {ok(s['forecast_passed'])}. Diuji pada {tanggal(start)} sampai {tanggal(end)}, meleset "
            f"{angka(s['forecast_mape'])}% (tebakan harga terakhir: {angka(s['naive_mape'])}%) dan "
            f"{angka(s['coverage'] * 100, 0)}% harga asli masuk rentang 80%. {spike_text}")


def main() -> None:
    df = quality.clean(quality.load_raw())
    wide = backtest.to_wide(df)
    level = spike.levels(wide).iloc[-1]
    series_files(df, wide, level)
    page = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__SUMMARY__", json.dumps(summary(df, wide, level)).replace("</", "<\\/"))
            .replace("__ALERT_TRACK__", alert_track(df, wide))
            .replace("__LIVE_TABLE__", live_table(df))
            .replace("__GATE__", gate_line()))
    (SITE / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {SITE / 'index.html'} and {len(list((SITE / 'series').glob('*.json')))} series files")


if __name__ == "__main__":
    main()
