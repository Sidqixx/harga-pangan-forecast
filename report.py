"""Build the public dashboard site/index.html (static; rebuilt by the daily job, served by GitHub Pages).

Run: python report.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

import backtest
import lebaran
import live
import quality
import spike
from model import HORIZON

SITE = Path(__file__).parent / "site"
NATIONAL = "Semua Provinsi"


def names(df: pd.DataFrame) -> pd.Series:
    return df.drop_duplicates("commodity_id").set_index("commodity_id")["commodity"]


def status(df: pd.DataFrame) -> pd.DataFrame:
    """Latest price of every active series vs its 4-week median and its commodity's alert level."""
    wide = backtest.to_wide(df)
    price = wide.ffill(limit=quality.STALE_DAYS).iloc[-1]
    theta = pd.Series(wide.columns.get_level_values("commodity_id").map(spike.THRESHOLDS), index=wide.columns)
    median = spike.levels(wide).iloc[-1] / (1 + theta)
    out = pd.DataFrame({
        "Komoditas": wide.columns.get_level_values("commodity_id").map(names(df)),
        "Provinsi": wide.columns.get_level_values("province"),
        "Tanggal": wide.apply(lambda s: s.last_valid_index()).dt.strftime("%d %b %Y").to_numpy(),
        "Harga (Rp/kg)": price.to_numpy(), "Median 4 minggu": median.to_numpy(),
        "Perubahan": (price / median - 1).to_numpy(), "Batas waspada": theta.to_numpy(),
    }).dropna()  # stopped or too-short series
    return out.sort_values("Perubahan", ascending=False, ignore_index=True)


def alert_section(df: pd.DataFrame) -> str:
    """Today's early warnings, and how past warnings turned out once their 14 days were over."""
    alerts = live.load(live.ALERTS)
    if alerts.empty:
        return "<p>Belum ada peringatan.</p>"
    today = alerts[alerts["origin"] == alerts["origin"].max()]
    hot = today[today["p"] >= spike.ALERT_CUTOFF].sort_values("p", ascending=False)
    table = pd.DataFrame({
        "Komoditas": hot["commodity_id"].map(names(df)), "Provinsi": hot["province"],
        "Harga (Rp/kg)": hot["price"], "Batas waspada (Rp/kg)": hot["level"], "Peluang melonjak": hot["p"],
    }).to_html(index=False, border=0, formatters={
        "Harga (Rp/kg)": "{:,.0f}".format, "Batas waspada (Rp/kg)": "{:,.0f}".format, "Peluang melonjak": "{:.0%}".format,
    })
    past = spike.score_alerts(alerts[alerts["p"] >= spike.ALERT_CUTOFF], backtest.to_wide(df))
    track = (f"Dari {len(past):,} peringatan yang 14 harinya sudah lewat, <b>{past['spike'].mean():.0%}</b> benar-benar melonjak."
             if len(past) else "Belum ada peringatan yang 14 harinya sudah lewat untuk dinilai.")
    return (f"<p>Deret yang hari ini masih normal tapi berpeluang &ge; {spike.ALERT_CUTOFF:.0%} menembus batas waspadanya "
            f"dalam 14 hari (data {today['origin'].max():%d %b %Y}): <b>{len(hot)}</b> deret. "
            f"Di backtest 2 tahun, 65% peringatan seperti ini benar-benar melonjak. {track}</p>\n{table}")


def gate_line() -> str:
    if not live.STATUS.exists():
        return ""
    s = json.loads(live.STATUS.read_text())
    verdict = "<b>lolos</b>" if s["passed"] else "<b>GAGAL</b>, forecast dan peringatan terakhir yang lolos tetap ditampilkan"
    return (f'<p class="muted">Gerbang rilis model {s["date"]}: {verdict}. Diuji ulang pada {s["holdout"]} yang tidak dilihat model: '
            f'error forecast {s["forecast_mape"]:.1f}% vs naive {s["naive_mape"]:.1f}%, rentang P10-P90 memuat {s["coverage"]:.0%} harga asli, '
            f'kualitas peringatan (average precision) {s["spike_ap"]:.2f} vs heuristik {s["heuristic_ap"]:.2f}.</p>')


def national_weekly(df: pd.DataFrame) -> pd.DataFrame:
    nat = df[df["province"] == NATIONAL]
    return nat.pivot(index="date", columns="commodity", values="price").resample("W").median()


def trend_fig(df: pd.DataFrame) -> go.Figure:
    weekly = national_weekly(df)
    fig = go.Figure([go.Scatter(x=weekly.index, y=weekly[c], name=c, mode="lines") for c in weekly])
    fig.update_layout(title="Harga rata-rata nasional (median mingguan)", yaxis_title="Rp/kg", hovermode="x unified")
    return fig


def region_fig(df: pd.DataFrame) -> go.Figure:
    """Per commodity: each province's weekly price vs the national price, last 52 weeks."""
    recent = df[df["date"] > df["date"].max() - pd.Timedelta(weeks=52)]
    commodities = sorted(recent["commodity"].unique())
    fig = go.Figure()
    for i, c in enumerate(commodities):
        w = recent[recent["commodity"] == c].pivot(index="date", columns="province", values="price").resample("W").median()
        gap = (w.drop(columns=NATIONAL).div(w[NATIONAL], axis=0) - 1) * 100
        gap = gap[gap.mean().sort_values().index]  # cheapest province at the bottom
        fig.add_heatmap(
            x=gap.index, y=gap.columns, z=gap.T.values, visible=i == 0, colorscale="RdBu_r", zmid=0,
            colorbar_title="% vs nasional", hovertemplate="%{y}<br>%{x|%d %b %Y}: %{z:+.0f}%<extra></extra>",
        )
    fig.update_layout(
        title="Selisih harga provinsi terhadap rata-rata nasional (52 minggu terakhir)", height=800,
        updatemenus=[{"x": 0, "y": 1.06, "xanchor": "left", "buttons": [
            {"label": c, "method": "update", "args": [{"visible": [j == i for j in range(len(commodities))]}]}
            for i, c in enumerate(commodities)
        ]}],
    )
    return fig


def lebaran_fig(df: pd.DataFrame) -> go.Figure:
    """National price around each Lebaran vs its level 45-60 days before, median over the years."""
    daily = df[df["province"] == NATIONAL].pivot(index="date", columns="commodity", values="price").asfreq("D").ffill()
    offsets = range(-30, 31)
    curves = []
    for day in lebaran.LEBARAN[lebaran.LEBARAN <= daily.index.max() - pd.Timedelta(days=30)]:
        base = daily.loc[day - pd.Timedelta(days=60):day - pd.Timedelta(days=45)].median()
        curves.append((daily.reindex([day + pd.Timedelta(days=o) for o in offsets]) / base - 1).set_axis(offsets) * 100)
    median = pd.concat(curves).groupby(level=0).median()
    fig = go.Figure([go.Scatter(x=median.index, y=median[c], name=c, mode="lines") for c in median])
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(
        title=f"Pola harga sekitar Lebaran (median {len(curves)} tahun)", hovermode="x unified",
        xaxis_title="Hari terhadap 1 Syawal", yaxis_title="% vs harga H-60 s.d. H-45",
    )
    return fig


def live_section(df: pd.DataFrame) -> str:
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


def html(df: pd.DataFrame) -> str:
    st = status(df)
    waspada = st[st["Perubahan"] > st["Batas waspada"]]
    table = waspada.to_html(index=False, border=0, formatters={
        "Harga (Rp/kg)": "{:,.0f}".format, "Median 4 minggu": "{:,.0f}".format,
        "Perubahan": "{:+.1%}".format, "Batas waspada": "+{:.0%}".format,
    })
    figs = "\n".join(f.to_html(full_html=False, include_plotlyjs=False) for f in (trend_fig(df), region_fig(df), lebaran_fig(df)))
    built = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pemantau Harga Pangan</title>
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
table {{ border-collapse: collapse; font-size: 14px; }} th, td {{ padding: 4px 10px; border-bottom: 1px solid #ddd; text-align: right; }}
th:nth-child(-n+3), td:nth-child(-n+3) {{ text-align: left; }} .muted {{ color: #666; font-size: 14px; }}
</style></head><body>
<h1>Pemantau Harga Pangan</h1>
<p class="muted">Harga harian pasar tradisional dari <a href="https://www.bi.go.id/hargapangan">PIHPS Bank Indonesia</a>,
data sampai <b>{df["date"].max():%d %b %Y}</b>. Diperbarui otomatis setiap hari ({built}).
<a href="https://github.com/Sidqixx/harga-pangan-forecast">Kode &amp; metodologi</a>.</p>
<h2>Status hari ini: waspada</h2>
<p><b>{len(waspada)}</b> dari {len(st)} deret harga sedang <b>waspada</b>: di atas median 4 minggu terakhirnya lebih dari
batas komoditasnya (beras 5%, cabai rawit merah 50%; batas per komoditas karena cabai wajar bergerak 3x lebih liar dari beras).</p>
{table if len(waspada) else ""}
<h2>Peringatan dini: lonjakan dalam 14 hari</h2>
{alert_section(df)}
<h2>Akurasi forecast (live)</h2>
{live_section(df)}
{gate_line()}
{figs}
</body></html>"""


def main() -> None:
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html(quality.clean(quality.load_raw())), encoding="utf-8")
    print(f"Wrote {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
