# ⚡ PJMW Energy Demand Forecast

An interactive **Streamlit dashboard** for exploring hourly electricity demand of the PJM West (PJMW) region and generating machine-learning forecasts with a tuned **XGBoost** model.

<!-- Add a screenshot here: ![Dashboard](docs/dashboard.png) -->

No file upload is needed — the app ships with a bundled dataset and a pre-trained model.

---

## Features

- **Dashboard** with four KPI cards: latest demand (with % change vs previous hour), average demand, peak demand, and model R².
- **Historical** tab: hourly demand line chart, demand distribution histogram, and average demand by day of week.
- **Forecast** tab: hourly forecast for up to 60 days, forecast summary (average / peak / low), day-by-day table, and an *Actual vs Predicted* model check.
- **Load Patterns** tab: hour × weekday heatmap, daily load shape by season, and monthly average demand.
- **Data** tab: browsable dataset, summary statistics, and XGBoost feature importance.
- **Control Panel** in the sidebar:
  - *Historical Period* — slider plus **From / To day** boxes, or **Show all data** (whole history).
  - *Forecast Horizon* — slider plus **From / To day** boxes (1–60 days, default 30).
- CSV downloads for the selected historical period, the forecast, and the dataset view.
- Long historical ranges are automatically averaged (3-hour → weekly) so charts stay fast; downloads always contain the raw hourly data.

---

## Project structure

```
pjmw_app/
├── app.py                          # the whole Streamlit app
├── PJMW_hourly_cleaned_edit.csv    # bundled hourly dataset
├── tuned_xgb_model.pkl             # pre-trained tuned XGBoost model
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # light theme + server settings
└── TrainedModels.ipynb             # (optional) notebook used to train the models
```

`app.py` looks for the CSV and the model in the same folder (also in `./data/` and `./models/`).

---

## Quick start

**Requirements:** Python 3.10 or newer.

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>.

### `requirements.txt`

```
streamlit>=1.50
pandas>=2.0
numpy>=1.26
plotly>=5.20
xgboost>=2.0
scikit-learn>=1.3
joblib>=1.3
```

---

## Data

| Item | Value |
|---|---|
| File | `PJMW_hourly_cleaned_edit.csv` |
| Coverage | 2002-04-01 → 2018-08-03 (≈ 5,967 days) |
| Rows | 143,202 hourly rows (4 duplicate timestamps are averaged on load) |
| Target | `PJMW_MW` — hourly demand in megawatts |

Columns: `Datetime`, `PJMW_MW`, `year`, `month`, `day`, `hour`, `Dayofweek` (0 = Monday), `isweekend`, `Is_Holiday`, `Season` (Winter = Dec–Feb, Spring = Mar–May, Summer = Jun–Aug, Fall = Sep–Nov).

> The dataset ends on **2018-08-03**, so "latest demand" and all forecasts start from that date — not from today's date.

---

## Model

**XGBoost regressor (tuned)** — `n_estimators=1000`, `learning_rate=0.1`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=1.0`, `random_state=42`.

**15 input features** (same order as in the notebook):

`hour`, `Dayofweek`, `isweekend`, `Is_Holiday`, `month`, `year`, `lag_1`, `lag_24`, `lag_168`, `rolling_24`, `rolling_168`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

- `lag_1 / lag_24 / lag_168` — demand 1 hour, 24 hours and 7 days earlier
- `rolling_24 / rolling_168` — mean of the previous 24 hours / 7 days
- `*_sin / *_cos` — cyclical encoding of hour of day and day of week

The notebook holds out the final year of data as the test set. `lag_1` is by far the most important feature.

---

## How the forecast works

The forecast is **recursive**, one hour at a time:

1. Start from the last real hour in the dataset.
2. Build the 15 features for the next hour (lags and rolling means come from real data first, then from earlier predictions).
3. Predict, append the prediction to the history, and repeat until the end of the chosen horizon.

Holidays in the forecast window use the **US federal holiday calendar**. *Day 1* is the first calendar day after the last data point.

### Accuracy — please read

| Evaluation | Result |
|---|---|
| One-step-ahead, last 30 days (the **R² card**) | R² ≈ 0.995, MAE ≈ 57 MW, MAPE ≈ 0.96 % |
| Recursive forecast, day 1 | ≈ 3–5 % MAPE |
| Recursive forecast, days 2–14 | ≈ 9 % MAPE |
| Recursive forecast, days 15–30 | ≈ 12 % MAPE (tends to under-predict by ~5 %) |
| Beyond 30 days | not backtested — treat as a rough trend |

The R² card measures accuracy when the model is given the *real* previous hour. A multi-day forecast has to rely on its own earlier predictions, so its error is much larger. Use long-horizon forecasts to understand the **trend**, not exact hourly values. (Backtest figures come from random start dates in the last year of data.)

---

## Using the dashboard

| Control | What it does |
|---|---|
| **Show all data** | Uses the entire history (2002 → 2018). Disables the range controls. |
| **Show last N days** slider | Shows the last N days of history. |
| **From day / To day** (historical) | Custom window measured in days back from the latest data. `0 → 30` = last 30 days; `100 → 400` = the period from 400 to 100 days ago. |
| **Select forecast days** slider | Forecasts up to day N (1–60). |
| **From day / To day** (forecast) | Days ahead. `1 → 30` = normal 30-day forecast; `31 → 45` = only days 31–45. |

Sliders and boxes stay in sync. If "From" is set higher than "To", it is corrected automatically. The sidebar can be closed with `«` and reopened with the `»` button at the top-left.

---

## Configuration

- **File names / paths** — constants `DATA_NAME` and `MODEL_NAME` at the top of `app.py`.
- **Features** — the `FEATURES` list in `app.py` must match the model's training features and order.
- **Forecast limit** — change the `60` in the *Select forecast days* slider and the two forecast number boxes in `sidebar()`.
- **Theme** — `.streamlit/config.toml`; extra styling lives in `inject_css()`.

---

## Updating the data or the model

**New data:** append rows to the CSV using the same columns (hourly, no gaps if possible), then restart the app or press `C` → *Clear cache*.

**New model:** retrain in `TrainedModels.ipynb`, then save it and replace the file:

```python
import joblib
joblib.dump(Tuned_xgb_model, "tuned_xgb_model.pkl")
```

Keep the same 15 features in the same order, or update `FEATURES` in `app.py`.

---

## Deploying (Streamlit Community Cloud)

1. Push the folder to a GitHub repository (the CSV and `.pkl` are about 8 MB each, which is fine for GitHub).
2. On <https://share.streamlit.io>, create a new app, choose the repo, and set the main file to `app.py`.
3. Streamlit installs everything from `requirements.txt` automatically.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not find PJMW_hourly_cleaned_edit.csv` (or the `.pkl`) | Put both files in the same folder as `app.py`. |
| XGBoost warning about loading an older serialized model | Harmless. To remove it, re-save the model with the XGBoost version you run (`model.get_booster().save_model("model.json")` or re-dump with joblib). |
| Sidebar disappeared | Click the `»` button at the top-left corner. |
| Fonts look different | The app loads the *Inter* font from Google Fonts; offline it falls back to the default sans-serif font. |
| Charts feel slow with "Show all data" | Expected on very long ranges — the line chart already averages the data; the histogram, heatmap and weekday chart use all rows. |
| Old results after replacing the CSV | Clear the cache (`C` → *Clear cache*) or restart Streamlit. |

---

## Limitations

- Demand only — no weather, temperature or economic inputs, which are the main drivers of real-world load.
- Recursive forecasts drift and under-predict on long horizons (see the accuracy table).
- The dataset ends in August 2018.
- The model is loaded from a pickle file — only use `.pkl` files you trust.

---

## Tech stack

Python · Streamlit · pandas · NumPy · XGBoost · scikit-learn · Plotly · joblib