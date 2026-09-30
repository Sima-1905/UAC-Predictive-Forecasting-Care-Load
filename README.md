# Predictive Forecasting of Care Load & Placement Demand

## Purpose
A Streamlit decision-support dashboard for forecasting **Children in HHS Care** and short-term **discharge/placement demand** from the supplied UAC daily dataset.

## Requirements covered
- Time-series preparation and daily continuity
- Trend/seasonality handling
- Lag features: t-1, t-7, t-14
- Rolling mean and standard deviation: 7 and 14 days
- Net Pressure = Transfers into HHS - Discharges
- Calendar features: day of week, month, day of year
- Strict chronological holdout
- Walk-forward validation
- Multi-horizon evaluation: 7, 14, 30 days
- Baselines: Naive Persistence, 7-Day Moving Average
- Statistical models: Exponential Smoothing, ARIMA, SARIMA
- ML models: Random Forest, Gradient Boosting
- MAE, RMSE, MAPE and horizon error
- Forecast uncertainty / 95% intervals
- Capacity breach probability signal
- Surge lead-time indicator
- Forecast Stability Index and Model Robustness Index
- Streamlit modules for care load, discharge demand, model comparison, uncertainty and scenarios

## Run locally
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Files
- `app.py` — Streamlit dashboard and forecasting pipeline
- `HHS_Unaccompanied_Alien_Children_Program (1).csv` — supplied dataset
- `UAC_Predictive_Forecasting_Project_Report.docx` — research/project report
- `requirements.txt` — dependencies

## Data-quality note
The supplied CSV contains completely blank rows and non-continuous reporting dates. Blank rows are removed. Missing calendar dates are inserted to create a continuous daily index and numeric gaps are time-interpolated for modeling/visualization. The dashboard reports this treatment explicitly; missing reporting days are not interpreted as zero activity.

## Important interpretation note
The dashboard is an academic decision-support implementation. Forecasts, scenario outputs and risk indicators are not guarantees of future operational conditions. Capacity breach probability is derived from the forecast interval under a normal approximation and should be treated as a risk signal, not an observed probability.
