"""Build the public dashboard site/index.html (static; rebuilt by the daily job, served by GitHub Pages).

Run: python report.py
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

import quality

SITE = Path(__file__).parent / "site"
NATIONAL = "Semua Provinsi"
SPIKE = 1.10  # price above 110% of its 4-week median = "waspada"; the same label the model will predict
# 1 Syawal as set by the government's sidang isbat
LEBARAN = pd.to_datetime([
    "2017-06-25", "2018-06-15", "2019-06-05", "2020-05-24", "2021-05-13",
    "2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31", "2026-03-21",
])


def status(df: pd.DataFrame) -> pd.DataFrame:
    """Latest price of every active series vs the median of its previous 20 reports (~4 weeks)."""
    latest = df["date"].max()
    rows = []
    for (commodity, province), s in df.groupby(["commodity", "province"]):
        if s["date"].iloc[-1] < latest - pd.Timedelta(days=quality.STALE_DAYS) or len(s) < 21:
            continue  # stopped reporting, or too short to compare
        median = s["price"].iloc[-21:-1].median()
        rows.append({
            "Komoditas": commodity, "Provinsi": province, "Tanggal": s["date"].iloc[-1].strftime("%d %b %Y"),
            "Harga (Rp/kg)": s["price"].iloc[-1], "Median 4 minggu": median,
            "Perubahan": s["price"].iloc[-1] / median - 1,
        })
    return pd.DataFrame(rows).sort_values("Perubahan", ascending=False, ignore_index=True)


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
    for day in LEBARAN[LEBARAN <= daily.index.max() - pd.Timedelta(days=30)]:
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


def html(df: pd.DataFrame) -> str:
    st = status(df)
    waspada = st[st["Perubahan"] > SPIKE - 1]
    table = st.head(20).to_html(index=False, border=0, formatters={
        "Harga (Rp/kg)": "{:,.0f}".format, "Median 4 minggu": "{:,.0f}".format, "Perubahan": "{:+.1%}".format,
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
<h2>Status hari ini</h2>
<p><b>{len(waspada)}</b> dari {len(st)} deret harga sedang <b>waspada</b>: lebih dari {SPIKE - 1:.0%} di atas median 4 minggu terakhirnya.
Tabel: 20 kenaikan tertinggi.</p>
{table}
{figs}
</body></html>"""


def main() -> None:
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html(quality.clean(quality.load_raw())), encoding="utf-8")
    print(f"Wrote {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
