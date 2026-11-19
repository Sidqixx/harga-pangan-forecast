# Harga Pangan: Food Price Monitor and Early Warning

**Live dashboard:** https://sidqixx.github.io/harga-pangan-forecast/ (rebuilt every day)

A daily-updated pipeline over Indonesia's official food price survey ([PIHPS, Bank Indonesia](https://www.bi.go.id/hargapangan)): traditional-market prices for 9 staple commodities in 34 provinces since 2017. The goal is an early warning for price spikes (*"which commodity in which province will jump in the next two weeks?"*) for the people who act on it: regional inflation teams (TPID) planning market operations, and traders deciding when to stock up.

The dashboard and UI are in Indonesian, since the users are Indonesian.

**Status:** all three releases are live: daily data pipeline and quality checks, a LightGBM forecaster that publishes 14-day forecasts with an 80% range, and a spike early warning, both behind a daily release gate and scored automatically against the prices that follow. See the [model card](MODEL_CARD.md).

## Pipeline

```
GitHub Actions, daily 10:00 WIB
  fetch.py    PIHPS API -> data/raw/YYYY-MM.csv   (re-fetches the last 14 days to catch late entries and revisions)
  quality.py  checks; the job fails (and emails) if the newest price is more than 10 days old
  live.py     release gate on the last 4 unseen weeks; if it passes, retrains on all data and saves
              14-day forecasts (P10/P50/P90 + naive) -> data/forecasts/, spike probabilities -> data/alerts/,
              gate result -> data/model_status.json; if it fails, publishes nothing and the job fails
  commit      only when something changed
  report.py   site/index.html -> GitHub Pages: status, early warnings, live accuracy, gate result
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

**1. Spikiness differs by two orders of magnitude, so one alert threshold cannot fit every commodity.** Annualized volatility of the national daily price ranges from 0.34 (beras) to 1.18 (cabai rawit merah). Using "more than 10% above its 4-week median" as the alert, a province-day is flagged 27.6% of the time for cabai rawit merah but 0.1% for beras. Alerting on 28% of days is noise, so the spike warning uses commodity-specific thresholds (release 3).

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

## Forecasting (release 2)

**Model:** one global LightGBM per quantile (P10, P50, P90) shared by all 304 province x commodity series. It predicts the log change from the last known price, so a Rp 13,000 rice series and a Rp 70,000 chili series share one scale, then converts back to rupiah. Features: price changes over 1 to 364 days, 28-day volatility, distance from the 28- and 91-day median, gap to the national price, national momentum, horizon, month, days until and since Lebaran, commodity and province (`model.py`).

**Backtest** (`backtest.py`): every Monday from Nov 2024 to Nov 2026 (104 origins) is a forecast origin. Each forecaster sees only prices up to the origin (the harness slices the history, so leakage is impossible by construction), predicts the next 14 calendar days, and is scored on the 314,768 days that had a real price. LightGBM is retrained every 6 months inside the backtest, using only examples whose target was already known.

MAPE (%), lower is better. Coverage = share of actual prices inside the P10–P90 range; 80% means the range is honest.

| Horizon | naive (last price) | seasonal (last year's move) | **LightGBM P50** | P10–P90 coverage |
|---|---|---|---|---|
| 1–3 days | 2.1 | 3.4 | **2.0** | 80.5% |
| 4–7 days | 4.0 | 6.3 | **3.7** | 79.8% |
| 8–14 days | 6.3 | 9.7 | **5.8** | 78.5% |

| 8–14 days ahead | naive | LightGBM | Error reduction |
|---|---|---|---|
| Bawang Merah | 5.7 | **4.8** | −16% |
| Cabai Rawit Merah | 13.6 | **12.1** | −11% |
| Cabai Merah Besar | 10.5 | **9.6** | −9% |
| Cabai Merah Keriting | 12.2 | **11.2** | −8% |
| Cabai Rawit Hijau | 9.2 | **8.6** | −7% |
| Beras / Telur / Ayam / Bawang Putih | 0.3 / 1.6 / 3.4 / 1.9 | 0.3 / 1.5 / 3.2 / 1.8 | 0% to −6% |

What this says:
- **"Same as the last price" is a strong baseline**, and the intuitive seasonal baseline loses to it everywhere: last year's moves do not repeat on the same dates.
- **LightGBM never loses to naive** in any commodity or horizon bucket, but the gains are modest and sit where volatility is: bawang merah and the chilies. Stable staples (beras, bawang putih) mostly tie, because there is almost nothing to win.
- **The P10–P90 range is honest overall (78.5–80.5%) but slightly too narrow for chilies 8–14 days ahead (73–78%)**: chili shocks are larger than the model expects. The spike warning uses its own classifier rather than this range, so alerts are not affected; it stays a known limitation in the model card.

## Spike early warning (release 3)

**Question:** which series that look normal today will jump above their alert level within 14 days? Series already above it are shown as *waspada*; the warning is about **new** spikes, the ones nobody has noticed yet.

**Alert level** = (1 + threshold) x the median of the last 20 reports. Thresholds are per commodity (beras 5% up to cabai rawit merah 50%), chosen on 2017–2024 data so a calm series turns into a spike within 14 days on about 10% of days. This follows finding 1: a single 10% rule would flag chilies on 28% of days and rice almost never.

**Model:** a LightGBM classifier on the forecaster's features plus today's price relative to the alert level (`spike.py`). A test checks that every feature at a date is unchanged when later data is added, so features computed once over the full history cannot leak.

**Backtest** (every weekday Nov 2024 – Nov 2026, 154,390 calm series-days, 7.5% became a spike):

| | Average precision |
|---|---|
| Random | 0.075 |
| Heuristic: the price closest to its alert level | 0.193 |
| **Classifier** | **0.490** |

It beats the heuristic for every commodity. At the cut-off used on the dashboard (p ≥ 0.5) there are about **9 alerts a day, 65% of which turn into a spike**, catching 27% of all new spikes; p ≥ 0.7 gives 2.7 alerts a day at 81% precision. The trade-off table and the limitations are in the [model card](MODEL_CARD.md).

**Release gate** (`live.py`, daily): the training recipe is re-run as if it were 4 weeks earlier and scored on the 4 weeks since. Forecasts and alerts are published only if the forecast does not lose to naive (2% tolerance), the P10–P90 range holds 70–90% of real prices, and the classifier beats the heuristic. Otherwise the last forecasts that passed stay online and the job fails visibly. First run (19 Nov 2026): forecast 3.2% vs naive 3.3%, coverage 84%, average precision 0.25 vs heuristic 0.10, passed.

## Running locally

Requires Python 3.14.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe fetch.py                       # last 14 days
.venv\Scripts\python.exe fetch.py --start 2017-01-01    # full backfill (~1 hour, the server is slow on purpose)
.venv\Scripts\python.exe quality.py
.venv\Scripts\python.exe backtest.py                    # baselines vs LightGBM over 2 years (~8 min)
.venv\Scripts\python.exe spike.py                       # spike classifier vs heuristic over 2 years (~1 min)
.venv\Scripts\python.exe live.py                        # gate, today's forecasts and alerts (~3 min)
.venv\Scripts\python.exe report.py                      # site/index.html
```

## Roadmap

1. ~~Daily data pipeline, quality checks, dashboard~~
2. ~~Forecast: naive and seasonal baselines, global LightGBM with quantiles, rolling-origin backtest, daily live forecasts scored against the prices that follow~~
3. ~~Spike early warning with commodity-specific thresholds, daily release gate, model card~~
4. Dashboard redesign: commodity and province filters instead of showing every series at once, clickable alert cards, one chart per selection with the forecast band attached, provinces as a ranked bar chart, and a lighter page (data as small JSON files, smaller chart library)

## Data

Price data © Bank Indonesia, PIHPS Nasional, retrieved from the public website. This repository is an independent project and is not affiliated with Bank Indonesia.
