# Model Card: Harga Pangan forecaster and spike early warning

## Models

| | Forecaster | Spike early warning |
|---|---|---|
| Task | Price for each of the next 14 calendar days, with an 80% range | Probability that a series that is calm today crosses its alert level within 14 days |
| Algorithm | LightGBM, quantile loss (P10, P50, P90), one global model for all series | LightGBM, binary log loss, one global model |
| Target | log(price on day t+h / last known price) | 1 if any reported price in (t, t+14] exceeds the alert level |
| Code | `model.py` | `spike.py` |

**Alert level** = (1 + threshold) x median of the series' last 20 reports (about 4 weeks). Thresholds are per commodity, chosen on 2017–2024 data so that a calm series turns into a spike within 14 days on about 10% of days: beras 5%, ayam / telur / bawang putih 10%, bawang merah 20%, cabai rawit hijau 30%, cabai merah besar / keriting 40%, cabai rawit merah 50%. One threshold for all would fire on almost every chili day and almost never on rice.

## Intended use

- Early warning for regional inflation teams (TPID), market operators and traders: which commodity in which province is likely to jump in the next two weeks, so supply or stock decisions can start earlier.
- A probability and a range to support a human decision, read together with local knowledge.

**Not intended for:** official price statistics, legal or contractual price setting, or automated trading. It is an independent project, not affiliated with Bank Indonesia.

## Data

PIHPS Nasional (Bank Indonesia), traditional-market daily prices, 9 commodities x 34 provinces + national average, Mar 2017 onwards, weekdays only. The 34 are PIHPS's provinces from before 2022; the four provinces split from Papua in 2022 are not reported separately by the source. 23 of 724,890 rows are removed by `quality.py` (typos and a test province). Series that stopped reporting for more than 10 days get no forecast and no alert.

Features (all computed from data up to the forecast date; a test asserts that adding later data does not change them): price changes over 1–364 days, 28-day volatility, distance from the 28- and 91-day median, gap and momentum vs the national price, horizon, month, days until and since Lebaran, commodity, province, and (warning only) today's price relative to the alert level.

## Evaluation

Rolling-origin backtest over Nov 2024 – Nov 2026; models are retrained every 6 months inside the backtest and only on examples whose outcome was already known.

**Forecaster** (104 Monday origins, 314,768 scored days), MAPE:

| Horizon | naive (last price) | LightGBM P50 | P10–P90 coverage |
|---|---|---|---|
| 1–3 days | 2.1% | 2.0% | 80.5% |
| 4–7 days | 4.0% | 3.7% | 79.8% |
| 8–14 days | 6.3% | 5.8% | 78.5% |

**Spike warning** (523 weekday origins, 154,390 calm series-days, 7.5% turned into a spike):

| | Average precision |
|---|---|
| Random | 0.075 |
| Heuristic: rank by how close the price already is to its alert level | 0.193 |
| **Classifier** | **0.490** |

| Alert cut-off | Alerts per day | Precision | Recall |
|---|---|---|---|
| p ≥ 0.3 | 21.3 | 49.9% | 48.1% |
| **p ≥ 0.5 (used)** | **9.3** | **64.7%** | **27.2%** |
| p ≥ 0.7 | 2.7 | 81.3% | 10.0% |

The cut-off of 0.5 keeps the daily list short enough to act on (about 9 series) with roughly two in three alerts coming true. Lower it to catch more spikes at the cost of more false alarms.

## Release gate and monitoring

Every day `live.py` re-runs the training recipe as if it were weeks earlier and scores it on the weeks since, which the model never saw. Forecasts and warnings are checked and published separately:

- **Forecasts** (last 4 weeks): published only if forecast MAPE ≤ naive MAPE x 1.02 and P10–P90 coverage is between 70% and 90%.
- **Warnings** (last 13 weeks): published only if the classifier's average precision beats the heuristic. Spikes are rare, so a shorter window can hold spikes in only a handful of series, and AP on so few events is noise: on 27 Nov 2026 the 4-week window had spikes in 7 series and the classifier "lost" (0.05 vs 0.08), while the 13-week window had 95 series and it won clearly (0.46 vs 0.16). If fewer than 20 series spiked in the window, the warning check is reported as not judged instead of failing.

A part that fails keeps its last published version online, the dashboard shows a notice, and the job fails (email). The result is saved to `data/model_status.json` and shown on the dashboard. Every published forecast and warning is scored against the real prices once they arrive, and the live accuracy is shown on the dashboard next to the backtest numbers.

## Limitations and risks

- **Precision is about 65%**: one in three alerts will not come true, and about 70% of spikes are not flagged at p ≥ 0.5. Sudden shocks with no lead-up in prices (floods, policy changes) are mostly missed.
- **The P10–P90 range is slightly too narrow for chilies 8–14 days ahead** (73–78% coverage in the backtest).
- **Rare events are thin**: beras and telur spikes are rare (1–2% of calm days), so their warnings rest on few examples.
- **Provinces with sparse reporting** (e.g. cabai merah besar outside Java) have less history and less reliable outputs.
- **Lebaran 2027** uses Muhammadiyah's date (9 March 2027) until the government's sidang isbat; update `lebaran.py` then.
- **Behavioral risk**: a public warning could, in principle, encourage hoarding. Warnings are probabilistic, shown with their track record, and aimed at people who manage supply rather than at consumers.

## Maintenance

- Retrained daily on all data, behind the release gate.
- Review thresholds yearly or if the share of calm days that turn into spikes drifts far from ~10%.
- Owner: Teuku Sidqi Aulia Rahmad (github.com/Sidqixx).
