# HDB resale price prediction: valuation model + Gradio app

Predicts the resale price of a Singapore HDB flat and turns it into a lender-view and a buyer-view valuation, with recent comparable sales alongside.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tantg88888/sit-ai-practice-2026/blob/main/week1-hdb-resale-price-prediction/notebook.ipynb)

**Team:** group capstone (Group 7), SIT ICT6001 Week 1.
My role: <TODO: fill in>

## Run the demo
1. Click **Open in Colab** above.
2. Keep the default **CPU** runtime (no GPU needed).
3. **Runtime → Run all.** `TRAIN = False` is the default.
4. After about **3–5 minutes**, a public Gradio link (`*.gradio.live`) appears at the bottom of the notebook. Pick a town, flat type, model, storey, street, floor area, lease year and month to get an estimate.

Demo mode skips the slow model race, ablations and tuning search, and shows their **recorded** results instead. It still retrains the final model live (about a minute on CPU) and recomputes the hold-out metrics. The data downloads automatically (see **Data**).

## Results
These are from the recorded run on the course snapshot (Jan 2017 → Jun 2026, 232,949 unique transactions).
The model trains on 2017-01 → 2024-12 and is tested once on the **future** period 2025-01 → 2026-06 (36,263 sales).

| Stage | Metric | Value |
|---|---|---|
| Model race (5-fold time-series CV) | RMSE: HistGB / RandomForest / Ridge | $52,840 / $62,186 / $72,331 |
| Tuning (3-fold time-series CV, 24 fits) | RMSE: baseline → tuned HistGB | $66,906 → $60,799 |
| **Final hold-out (2025-01 → 2026-06)** | **RMSE / MAE / R²** | **$64,269 / $49,418 / 0.903** |
| Hold-out bias | Share of flats under-predicted | 87.8% (mean signed error −$44,359) |
| Safety margin −3% (lender view) | Over-predicted flats | 12.2% → 5.2% |

## How it works
- **Leakage-safe pipeline:** scikit-learn `Pipeline` + `ColumnTransformer` (median impute + scale, one-hot for town / flat type / flat model, target encoding for `street_name` fitted per fold), wrapped in a log-price `TransformedTargetRegressor`.
- **Features:** remaining lease parsed to months, storey-band midpoint, flat age, transaction year and month. `price_per_sqm` is deliberately excluded because it would leak the target.
- **Time-based evaluation:** `TimeSeriesSplit` CV on the training years, plus a frozen future hold-out that is used only once.
- **Champion and ablations:** HistGradientBoosting beat RandomForest and Ridge. Four one-variable ablations were run: dropping target encoding hurt; dropping the lease feature was a near-tie; tuning helped; and the raw target was marginally better than log on CV (65,924 vs 66,906) but the log target was kept (reasons in the notebook).
- **Failure analysis and decision:** the worst errors are under-predictions of rare million-dollar flats. Because over-valuation is the costlier error for a lender, the app shows a **−3% lender view** and a **+3% buyer-advice view**.
- **App:** Gradio form → one raw row built in the dataset's format (including the `remaining_lease` text) → the same `engineer()` + pipeline → estimate, margin views and the 10 most recent sales of the same flat type in the same town. For the app, the model is refit on all transactions.

## Reproduce training
1. Open the notebook in Colab (CPU is fine).
2. Set `TRAIN = True` in the first code cell.
3. Run all. It takes about 5–10 minutes on a free Colab CPU and reruns the model race, the four ablations and the `RandomizedSearchCV` tuning.

No model files are published. The final model retrains in about a minute, so the notebook rebuilds it on every run.

## Data
- **Source:** *Resale flat prices based on registration date from Jan-2017 onwards*, Housing & Development Board, via [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view), under the [Singapore Open Data Licence v1.0](https://data.gov.sg/open-data-licence).
- **Load order in the notebook:**
  1. A local `hdb_resale_flat_prices.csv`.
  2. The course snapshot in this repo at `data/hdb_resale_flat_prices.csv`. **The owner must upload this file**; it is about 20 MB.
  3. The live data.gov.sg API (poll / initiate download).
- **Live data caveat:** the live data includes months after Jun 2026, so its numbers will differ slightly from the table above. The hold-out then simply runs from 2025-01 to the latest month.

## Limitations
- The model systematically **under-predicts** in a rising market: 87.8% of hold-out flats came in below their actual price. Million-dollar flats and rare types (MULTI-GENERATION, 1 ROOM) have the largest errors.
- Only the lease **start year** is available, so the app estimates remaining lease from the dataset's typical lease length for that year. It can be off by a few months.
- Tree models cannot extrapolate. For months after the latest data, the price level is held flat.
- Block-level premiums (view, heritage, proximity to MRT) are not modelled. Location goes only down to street level.
- This is a learning project, not a valuation service. Its estimates are not financial advice.
