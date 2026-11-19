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
    curves, years = lebaran_curves(df)
    out = SITE / "series"
    out.mkdir(parents=True, exist_ok=True)
    for cid in wide.columns.unique("commodity_id"):
        provinces = wide[cid].columns
        f = forecast[forecast["commodity_id"] == cid]
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
        }
        (out / f"{cid}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def summary(df: pd.DataFrame, wide: pd.DataFrame, level: pd.Series) -> dict:
    st = status(df, wide, level)
    waspada = st[st["change"] > st["threshold"]].sort_values("change", ascending=False)
    alerts = latest(live.ALERTS)
    hot = alerts[alerts["p"] >= spike.ALERT_CUTOFF].sort_values("p", ascending=False) if len(alerts) else alerts
    n = names(df)
    return {
        "dataDate": f"{df['date'].max():%d %b %Y}",
        "built": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
        "national": NATIONAL,
        "commodities": [{"id": c, "name": n[c], "threshold": spike.THRESHOLDS[c]} for c in sorted(n.index, key=n.get)],
        "series": len(st),
        "waspada": [{"cid": r.cid, "province": r.province, "price": int(r.price), "change": round(r.change, 3),
                     "threshold": r.threshold} for r in waspada.itertuples()],
        "alerts": [{"cid": r.commodity_id, "province": r.province, "price": int(r.price), "level": int(r.level), "p": r.p}
                   for r in hot.itertuples()],
        "alertCutoff": spike.ALERT_CUTOFF,
    }


def alert_track(df: pd.DataFrame, wide: pd.DataFrame) -> str:
    alerts = live.load(live.ALERTS)
    past = spike.score_alerts(alerts[alerts["p"] >= spike.ALERT_CUTOFF], wide) if len(alerts) else alerts
    if not len(past):
        return "Rekam jejak live muncul setelah peringatan pertama lewat 14 hari."
    return f"Live: dari {len(past):,} peringatan yang 14 harinya sudah lewat, <b>{past['spike'].mean():.0%}</b> benar-benar melonjak."


def live_table(df: pd.DataFrame) -> str:
    """Accuracy of the daily forecasts so far, scored against prices that have since been reported."""
    s = live.score(df)
    if s.empty:
        return "<p>Belum ada forecast yang bisa dinilai: harga aslinya belum keluar.</p>"
    s = s.assign(horizon=pd.cut(s["h"], [0, 3, 7, HORIZON], labels=["1-3 hari", "4-7 hari", "8-14 hari"]))
    table = pd.DataFrame([
        {
            "Horizon": hz, "Dinilai": len(g),
            "Error naive (harga terakhir)": backtest.mape(g["naive"], g["actual"]),
            "Error model (P50)": backtest.mape(g["p50"], g["actual"]),
            "Harga asli di rentang P10-P90": g["actual"].between(g["p10"], g["p90"]).mean(),
        }
        for hz, g in s.groupby("horizon", observed=True)
    ]).to_html(index=False, border=0, formatters={
        "Dinilai": "{:,}".format, "Error naive (harga terakhir)": "{:.1f}%".format,
        "Error model (P50)": "{:.1f}%".format, "Harga asli di rentang P10-P90": "{:.0%}".format,
    })
    return (f"<p>Forecast harian berjalan sejak <b>{s['origin'].min():%d %b %Y}</b>. "
            f"{len(s):,} prediksi sudah dinilai dengan harga asli (error = MAPE, makin kecil makin baik; "
            f"rentang P10-P90 yang jujur memuat sekitar 80% harga asli).</p>\n{table}")


def gate_line() -> str:
    if not live.STATUS.exists():
        return ""
    s = json.loads(live.STATUS.read_text())
    verdict = "<b>lolos</b>" if s["passed"] else "<b>GAGAL</b>, forecast dan peringatan terakhir yang lolos tetap ditampilkan"
    return (f'Gerbang rilis model {s["date"]}: {verdict}. Diuji ulang pada {s["holdout"]} yang tidak dilihat model: '
            f'error forecast {s["forecast_mape"]:.1f}% vs naive {s["naive_mape"]:.1f}%, rentang P10-P90 memuat {s["coverage"]:.0%} harga asli, '
            f'kualitas peringatan (average precision) {s["spike_ap"]:.2f} vs heuristik {s["heuristic_ap"]:.2f}.')


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
