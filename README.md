# Harga Pangan: Food Price Monitor and Early Warning

**Live dashboard:** https://sidqixx.github.io/harga-pangan-forecast/ (rebuilt every day)

A daily-updated pipeline over Indonesia's official food price survey ([PIHPS, Bank Indonesia](https://www.bi.go.id/hargapangan)): traditional-market prices for 9 staple commodities in 34 provinces since 2017. The goal is an early warning for price spikes (*"which commodity in which province will jump in the next two weeks?"*) for the people who act on it: regional inflation teams (TPID) planning market operations, and traders deciding when to stock up.

The dashboard and UI are in Indonesian, since the users are Indonesian.

**Status:** release 1 (data pipeline, quality checks, dashboard) is live. Forecasting and the spike classifier come next.

## Pipeline

```
GitHub Actions, daily 10:00 WIB
  fetch.py    PIHPS API -> data/raw/YYYY-MM.csv   (re-fetches the last 14 days to catch late entries and revisions)
  quality.py  checks; the job fails (and emails) if the newest price is more than 10 days old
  commit      only when prices changed
  report.py   site/index.html -> GitHub Pages
```

- **Idempotent:** rows are upserted on `(date, province, commodity_id)`; running twice changes nothing, and a revised price replaces the old one.
- **Polite and resilient:** one request per ~quarter, 5 s between requests, 6 attempts with exponential backoff. The server resets connections under load (101 retries during the backfill, all recovered). A failed chunk does not stop the others; the job still saves what it got, then fails visibly with the exact command to re-run.
- **Storage:** one CSV per month, so the daily job only rewrites the current month (small diffs in git). 725K rows / 44 MB in total, which is small data: pandas and DuckDB handle it on a laptop, so there is no warehouse or Spark.

## Data quality

| Check | Rule | Found |
|---|---|---|
| Unknown province | name not in the 34 provinces + national average | 6 rows of a test province named `Dum` (15 Dec 2021) |
| Outlier | price outside 0.4x to 2.5x the median of the 21 surrounding reports of the same series | 17 rows, e.g. beras at Rp 0, ayam at Rp 1,100, bawang merah at Rp 1,200 |
| Stale data | newest price older than 10 days (Lebaran leave can pause reporting for about a week) | none |

Only 23 of 724,890 rows (0.003%) are dropped. The outlier rule compares against a **centered** median on purpose: a typo is a one-day blip, while a real shock (cabai rawit hijau in Sumatera Utara tripled in Dec 2025 and stayed there for weeks) moves the median with it and is kept. Real spikes are exactly what the model must learn, so the cleaning must not remove them. Both cases are covered by tests.

Seven series stopped reporting (mostly cabai merah besar: Gorontalo 2020, Bangka Belitung 2022, Sulawesi Utara 2023, Maluku Jul 2026) and are excluded from the live status table.

## Findings (data up to 19 Nov 2026)

**1. Volatility differs by an order of magnitude, so one alert threshold cannot fit every commodity.** Annualized volatility of the national daily price ranges from 0.34 (beras) to 1.18 (cabai rawit merah). Using "more than 10% above its 4-week median" as the alert, a province-day is flagged 27.6% of the time for cabai rawit merah but 0.1% for beras. Alerting on 28% of days is noise, so the spike label needs commodity-specific thresholds (a decision for release 2).

| Commodity | Annualized volatility | Province-days >10% above 4-week median |
|---|---|---|
| Beras Kualitas Medium I | 0.34 | 0.1% |
| Daging Ayam Ras Segar | 0.43 | 5.8% |
| Bawang Putih | 0.62 | 5.4% |
| Bawang Merah | 0.65 | 14.7% |
| Telur Ayam Ras | 0.68 | 3.2% |
| Cabai Rawit Hijau | 0.93 | 20.1% |
| Cabai Merah Keriting | 0.99 | 24.8% |
| Cabai Merah Besar | 1.12 | 23.2% |
| Cabai Rawit Merah | 1.18 | 27.6% |

**2. The Lebaran effect is a sharp, late jump, not a slow build-up.** National price vs its level 45 to 60 days before 1 Syawal, median over the 10 Lebaran holidays since 2017 (the dashboard chart). Daging ayam is the clearest Lebaran item: +4.2% a week before, +13.2% on the day before. Cabai is *below* its earlier level in the last week (cabai rawit merah −21.2%, cabai merah besar −8.2% at H-7) and then jumps within days (to +2.4% and +8.7% at H-1). Telur is up about 5% throughout, bawang merah keeps rising after Lebaran (+11.0% at H+30), and beras does not move (±0.2%). A jump that happens in the last few days is exactly what a two-week warning has to anticipate.

**3. Eastern Indonesia pays far more, and the gap is persistent.** Over the last 30 days, the most expensive province pays 5.0x the cheapest for cabai rawit hijau, but only 1.5x for beras. Maluku Utara is the most expensive province for 5 of the 9 commodities and Papua for 2; Nusa Tenggara Barat is the cheapest for 5. The dashboard heatmap shows the gap holding all year, so province is a strong feature for the model.

**4. Clear seasonality outside Lebaran.** Bawang merah sits 13% to 16% above its yearly average in May and June and 14% to 17% below it in September and October. Cabai rawit merah is cheapest from September to November (17% to 21% below average).

## Running locally

Requires Python 3.14.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe fetch.py                       # last 14 days
.venv\Scripts\python.exe fetch.py --start 2017-01-01    # full backfill (~1 hour, the server is slow on purpose)
.venv\Scripts\python.exe quality.py
.venv\Scripts\python.exe report.py                      # site/index.html
```

## Roadmap

1. ~~Daily data pipeline, quality checks, dashboard~~
2. Forecast: naive and seasonal baselines, then a global LightGBM with quantiles (P10/P50/P90) over all province x commodity series, rolling-origin backtests, and live scoring of every past forecast
3. Spike warning: probability of a commodity-specific spike within 14 days, retraining with a champion/challenger gate, model card
4. Dashboard redesign once forecasts exist: commodity and province filters instead of showing every series at once, clickable alert cards, one chart per selection with the forecast band attached, provinces as a ranked bar chart, and a lighter page (data as small JSON files, smaller chart library)

## Data

Price data © Bank Indonesia, PIHPS Nasional, retrieved from the public website. This repository is an independent project and is not affiliated with Bank Indonesia.
