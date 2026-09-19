# 🚦 Junction Traffic Volume Predictor

> Predicts hourly vehicle counts at 4 road junctions using LightGBM — trained on 48,120 hourly records (2015–2017).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## Live Demo

**[your-app-name.streamlit.app](https://your-app-name.streamlit.app)** ← replace with your actual URL after deployment

Select a junction (1–4), pick a date and hour, and get an instant traffic volume prediction with a 24-hour forecast chart.

---

## Project Overview

| | |
|---|---|
| **Goal** | Forecast hourly vehicle count at road junctions |
| **Dataset** | 2015–2017 historical traffic data, 4 junctions, 48,120 records |
| **Best model** | LightGBM (refined with GridSearchCV + feature selection) |
| **R²** | 0.9699 |
| **MAE** | 3.28 vehicles/hour |
| **MAPE** | 22.19% (inflated by near-zero overnight counts) |

---

## Repository Structure

```
uber_traffic_prediction/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   └── predict.py                  # Feature engineering + model inference
│
├── models/
│   ├── FINAL_lgb_junction1.pkl     # Junction-specific LightGBM models
│   ├── FINAL_lgb_junction2.pkl
│   ├── FINAL_lgb_junction3.pkl
│   ├── FINAL_lgb_junction4.pkl
│   └── FINAL_lightgbm_model.pkl    # Global fallback model
│
├── notebooks/
│   ├── EDA_n_Data_Preparation.ipynb
│   ├── Data_Visualization_and_Analysis.ipynb
│   └── Data_Modeling_n_Model_evaluation.ipynb
│
└── data/                           # git-ignored if large
    └── (dataset CSV files)
```

---

## Feature Engineering

The model uses these feature groups:

| Group | Features |
|---|---|
| Temporal | hour, hour_sin, hour_cos, day_of_week, dow_sin, dow_cos, month, year |
| Junction | junction (1–4) |
| Calendar | is_weekend, is_workday, is_rush_hour, is_holiday, is_special_event |
| Weather | temp_norm, wspd_norm, pres, pres_norm |
| Time encoding | time_of_day_enc (0=Night … 4=Late Night) |
| Lag features | lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_168h |
| Rolling | rolling_mean/std/max/min 3h/6h/24h, ewma_12h, ewma_24h |

---

## Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| LightGBM (refined) | **3.28** | **4.64** | **0.9699** |
| XGBoost | 3.91 | 5.30 | 0.956 |
| Random Forest | 4.12 | 5.71 | 0.948 |
| ARIMA (baseline) | — | — | — |

---

## Run Locally

```bash
git clone https://github.com/yourusername/uber_traffic_prediction.git
cd uber_traffic_prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Place your .pkl model files in models/
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (include `models/` folder)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch `main`, main file `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## Notebooks

| Notebook | Contents |
|---|---|
| EDA & Data Preparation | Missing value handling, outlier capping (IQR), type corrections |
| Data Visualization & Analysis | Traffic patterns by hour, day, junction, weather correlation |
| Modeling & Evaluation | LightGBM / XGBoost / RF / ARIMA, TimeSeriesSplit CV, feature importance, refinement |

---

## Author

**Sayed Shoaib**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)
