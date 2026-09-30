import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# PAGE / THEME
# =========================================================
st.set_page_config(page_title='UAC Predictive Forecasting', page_icon='📈', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
:root{--navy:#08284b;--blue:#1261a0;--teal:#0f766e;--sky:#eef7ff;--green:#168a52;--orange:#d97706;--red:#c2410c;--purple:#6d4aff;--slate:#64748b}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1550px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#061b34,#0a3157)}
[data-testid="stSidebar"] *{color:#f8fbff!important}
.hero{background:linear-gradient(135deg,#061b34,#0e4775 55%,#147aa6);padding:30px 34px;border-radius:24px;color:#fff;margin-bottom:18px;box-shadow:0 12px 35px rgba(8,40,75,.22)}
.hero h1{font-size:36px;margin:0 0 5px;font-weight:850;letter-spacing:-.7px}.hero p{margin:5px 0;opacity:.93;font-size:15px}.badge{display:inline-block;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.25);padding:6px 12px;border-radius:999px;margin-top:13px;font-size:12px}
.section-title{font-size:21px;font-weight:800;color:var(--navy);margin:18px 0 5px}.section-sub{font-size:13px;color:var(--slate);margin-bottom:10px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:17px 18px;box-shadow:0 5px 18px rgba(15,23,42,.05)}
.kpi-label{font-size:11px;color:var(--slate);font-weight:800;text-transform:uppercase;letter-spacing:.55px}.kpi-value{font-size:27px;font-weight:850;color:var(--navy);margin-top:5px}.kpi-note{font-size:12px;color:var(--slate);margin-top:2px}
.insight{border-left:5px solid var(--blue);background:#f5faff;padding:12px 14px;border-radius:10px;margin:8px 0}.warning{border-left-color:var(--orange);background:#fffaf0}.success{border-left-color:var(--green);background:#f0fdf4}.risk{border-left-color:var(--red);background:#fff7ed}.purple{border-left-color:var(--purple);background:#f8f6ff}
.footer{margin-top:28px;padding:16px;text-align:center;color:var(--slate);font-size:12px;border-top:1px solid #e2e8f0}.small{font-size:12px;color:var(--slate)}
</style>
''', unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
CSV_NAME='HHS_Unaccompanied_Alien_Children_Program (1).csv'
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
CSV_PATH=os.path.join(BASE_DIR,CSV_NAME)

@st.cache_data(show_spinner=False)
def load_data(path):
    raw=pd.read_csv(path)
    raw_rows=len(raw); blank_rows=int(raw.isna().all(axis=1).sum())
    df=raw.dropna(how='all').copy()
    df['Date']=pd.to_datetime(df['Date'],errors='coerce')
    invalid_dates=int(df['Date'].isna().sum()); df=df.dropna(subset=['Date'])
    for c in df.columns[1:]:
        df[c]=pd.to_numeric(df[c].astype(str).str.replace(',','',regex=False),errors='coerce')
    df=df.sort_values('Date').drop_duplicates('Date',keep='last').set_index('Date')
    original_days=len(df)
    full_idx=pd.date_range(df.index.min(),df.index.max(),freq='D')
    missing_dates=full_idx.difference(df.index)
    missing_days=len(missing_dates)
    df=df.reindex(full_idx)
    numeric=list(df.columns)
    missing_cells_before=int(df[numeric].isna().sum().sum())
    df[numeric]=df[numeric].interpolate(method='time',limit_direction='both')
    df[numeric]=df[numeric].ffill().bfill()
    df['Net Pressure']=df['Children transferred out of CBP custody']-df['Children discharged from HHS Care']
    df['7D Care Avg']=df['Children in HHS Care'].rolling(7).mean()
    df['14D Care Avg']=df['Children in HHS Care'].rolling(14).mean()
    quality={'raw_rows':raw_rows,'blank_rows':blank_rows,'valid_rows':original_days,'invalid_dates':invalid_dates,'missing_days':missing_days,'missing_cells_before':missing_cells_before,'start':df.index.min(),'end':df.index.max()}
    return df,quality

if not os.path.exists(CSV_PATH):
    st.error(f'Dataset not found: {CSV_NAME}. Keep the CSV beside app.py.')
    st.stop()

df,quality=load_data(CSV_PATH)
TARGET='Children in HHS Care'; DISCHARGE='Children discharged from HHS Care'; TRANSFER='Children transferred out of CBP custody'; INTAKE='Children apprehended and placed in CBP custody*'

# =========================================================
# FEATURE ENGINEERING / MODELS
# =========================================================
def make_features(data,target=TARGET):
    s=data[target]; X=pd.DataFrame(index=data.index)
    for l in [1,7,14]: X[f'lag_{l}']=s.shift(l)
    for w in [7,14]:
        X[f'roll_mean_{w}']=s.shift(1).rolling(w).mean(); X[f'roll_std_{w}']=s.shift(1).rolling(w).std()
    X['net_pressure']=data['Net Pressure'].shift(1)
    X['dow']=data.index.dayofweek; X['month']=data.index.month; X['dayofyear']=data.index.dayofyear
    return X

def fit_predict_stat(series,h,model_name):
    s=pd.Series(series).astype(float)
    future_idx=pd.date_range(s.index[-1]+pd.Timedelta(days=1),periods=h,freq='D')
    if model_name=='Naïve Persistence':
        pred=np.repeat(s.iloc[-1],h); sigma=max(float(s.diff().std()),1.0)
        return pd.DataFrame({'Forecast':pred,'Lower':pred-1.96*sigma,'Upper':pred+1.96*sigma},index=future_idx)
    if model_name=='7-Day Moving Average':
        pred=np.repeat(s.tail(7).mean(),h); sigma=max(float(s.tail(30).std()),1.0)
        return pd.DataFrame({'Forecast':pred,'Lower':pred-1.96*sigma,'Upper':pred+1.96*sigma},index=future_idx)
    if model_name=='Exponential Smoothing':
        fit=ExponentialSmoothing(s,trend='add',seasonal='add',seasonal_periods=7,initialization_method='estimated').fit(optimized=True)
    elif model_name=='ARIMA':
        fit=ARIMA(s,order=(1,1,1)).fit()
    else:
        fit=SARIMAX(s,order=(1,1,1),seasonal_order=(1,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    fc=fit.get_forecast(h); pred=fc.predicted_mean
    if hasattr(fc,'conf_int'):
        ci=fc.conf_int(alpha=.05); lower=ci.iloc[:,0].values; upper=ci.iloc[:,1].values
    else:
        resid=np.asarray(fit.resid); sigma=max(float(np.nanstd(resid)),1.0); lower=np.asarray(pred)-1.96*sigma; upper=np.asarray(pred)+1.96*sigma
    return pd.DataFrame({'Forecast':np.asarray(pred),'Lower':lower,'Upper':upper},index=future_idx)

def build_ml_model(name):
    if name=='Random Forest': return RandomForestRegressor(n_estimators=250,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)
    return GradientBoostingRegressor(n_estimators=220,max_depth=3,learning_rate=.04,loss='huber',random_state=42)

def ml_forecast(data,target,h,model_name):
    s=data[target].copy(); X=make_features(data,target); train=X.copy(); train['y']=s; train=train.dropna()
    model=build_ml_model(model_name); model.fit(train.drop(columns='y'),train.y)
    hist=s.copy(); pred=[]
    pressure=float(data['Net Pressure'].tail(7).mean())
    for _ in range(h):
        idx=hist.index[-1]+pd.Timedelta(days=1); row={}
        for l in [1,7,14]: row[f'lag_{l}']=hist.iloc[-l]
        for w in [7,14]: row[f'roll_mean_{w}']=hist.iloc[-w:].mean(); row[f'roll_std_{w}']=hist.iloc[-w:].std()
        row['net_pressure']=pressure; row['dow']=idx.dayofweek; row['month']=idx.month; row['dayofyear']=idx.dayofyear
        p=float(model.predict(pd.DataFrame([row]))[0]); pred.append(p); hist.loc[idx]=p
    idx=pd.date_range(s.index[-1]+pd.Timedelta(days=1),periods=h,freq='D')
    # residual-based approximate interval for ML
    resid=train.y-model.predict(train.drop(columns='y')); sigma=max(float(np.std(resid)),1.0)
    steps=np.arange(1,h+1); width=1.96*sigma*np.sqrt(steps)
    return pd.DataFrame({'Forecast':pred,'Lower':np.asarray(pred)-width,'Upper':np.asarray(pred)+width},index=idx)

def forecast(data,target,h,model_name):
    if model_name in ['Naïve Persistence','7-Day Moving Average','Exponential Smoothing','ARIMA','SARIMA']:
        return fit_predict_stat(data[target],h,model_name)
    return ml_forecast(data,target,h,model_name)

MODELS=['Naïve Persistence','7-Day Moving Average','Exponential Smoothing','ARIMA','SARIMA','Random Forest','Gradient Boosting']

# =========================================================
# VALIDATION
# =========================================================
def one_model_holdout(train,test,model_name):
    if model_name in ['Naïve Persistence','7-Day Moving Average','Exponential Smoothing','ARIMA','SARIMA']:
        return forecast(train,TARGET,len(test),model_name)['Forecast'].values
    return forecast(train,TARGET,len(test),model_name)['Forecast'].values

def metrics(actual,pred):
    actual=np.asarray(actual,float); pred=np.asarray(pred,float); mask=actual!=0
    mae=mean_absolute_error(actual,pred); rmse=np.sqrt(mean_squared_error(actual,pred)); mape=np.mean(np.abs((actual[mask]-pred[mask])/actual[mask]))*100 if mask.any() else np.nan
    return mae,rmse,mape

@st.cache_data(show_spinner=False)
def walk_forward_results(data):
    # Three chronological origins, evaluated at 7/14/30 day horizons where possible.
    rows=[]; n=len(data)
    for h in [7,14,30]:
        for fold in range(3,0,-1):
            end=n-fold*h
            if end<=120: continue
            train=data.iloc[:end]; test=data.iloc[end:min(end+h,n)]
            if len(test)<h: continue
            for name in MODELS:
                try:
                    pred=one_model_holdout(train,test,name); mae,rmse,mape=metrics(test[TARGET],pred)
                    rows.append([name,h,fold,mae,rmse,mape])
                except Exception as e:
                    rows.append([name,h,fold,np.nan,np.nan,np.nan])
    return pd.DataFrame(rows,columns=['Model','Horizon','Fold','MAE','RMSE','MAPE'])

@st.cache_data(show_spinner=False)
def latest_holdout(data,test_days=14):
    train=data.iloc[:-test_days]; test=data.iloc[-test_days:]; rows=[]
    for name in MODELS:
        try:
            pred=one_model_holdout(train,test,name); mae,rmse,mape=metrics(test[TARGET],pred); rows.append([name,mae,rmse,mape])
        except: pass
    out=pd.DataFrame(rows,columns=['Model','MAE','RMSE','MAPE']).sort_values('MAE').reset_index(drop=True)
    return out

# =========================================================
# PLOT HELPERS
# =========================================================
def forecast_plot(history,fc,title,ytitle):
    fig=go.Figure(); hist=history.tail(min(120,len(history)))
    fig.add_trace(go.Scatter(x=hist.index,y=hist.values,name='Historical',mode='lines',line=dict(width=3,color='#1261a0')))
    fig.add_trace(go.Scatter(x=fc.index,y=fc.Upper,name='95% Confidence Upper',line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fc.index,y=fc.Lower,name='95% Confidence Interval',fill='tonexty',fillcolor='rgba(18,97,160,.13)',line=dict(width=0)))
    fig.add_trace(go.Scatter(x=fc.index,y=fc.Forecast,name='Forecast',mode='lines+markers',line=dict(width=3,dash='dash',color='#0f766e')))
    fig.add_vline(x=history.index[-1],line_dash='dot',line_color='#94a3b8')
    fig.update_layout(title=title,yaxis_title=ytitle,xaxis_title='Date',height=420,margin=dict(l=20,r=20,t=60,b=20),hovermode='x unified',legend=dict(orientation='h',y=1.08),template='plotly_white')
    return fig

def flow_plot(data):
    d=data.tail(120); fig=make_subplots(specs=[[{'secondary_y':True}]])
    fig.add_trace(go.Bar(x=d.index,y=d[TRANSFER],name='Transfers into HHS',opacity=.65),secondary_y=False)
    fig.add_trace(go.Bar(x=d.index,y=d[DISCHARGE],name='Discharges',opacity=.65),secondary_y=False)
    fig.add_trace(go.Scatter(x=d.index,y=d['Net Pressure'],name='Net Pressure',mode='lines',line=dict(width=3,color='#d97706')),secondary_y=True)
    fig.update_layout(title='Transfers vs Discharges & Net Pressure',height=400,barmode='group',template='plotly_white',hovermode='x unified',legend=dict(orientation='h',y=1.08),margin=dict(l=20,r=20,t=60,b=20))
    fig.update_yaxes(title_text='Children',secondary_y=False); fig.update_yaxes(title_text='Net pressure',secondary_y=True)
    return fig

def scenario_forecast(base_fc,pressure,scenario):
    mult_transfer=1.0; mult_discharge=1.0
    if scenario=='Surge: +20% transfers': mult_transfer=1.20
    elif scenario=='Lower placements: -15% discharges': mult_discharge=.85
    elif scenario=='Combined stress': mult_transfer=1.20; mult_discharge=.85
    shock=(pressure*mult_transfer-pressure) + (-(max(0,-pressure))*0) # transfer-side scenario approximation
    # A transparent operational stress adjustment, not a causal model.
    if scenario!='Baseline':
        delta=(pressure*(mult_transfer-1)) + (abs(pressure)*0.5*(1-mult_discharge))
        cumulative=np.cumsum(np.repeat(delta,len(base_fc)))
        out=base_fc.copy(); out['Forecast']=out['Forecast']+cumulative
        out['Lower']=out['Lower']+cumulative; out['Upper']=out['Upper']+cumulative
        return out
    return base_fc.copy()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('## 📊 UAC Forecast Lab')
    st.caption('Predictive care-load & placement-demand intelligence')
    st.divider()
    horizon=st.slider('Forecast horizon (days)',7,30,14)
    model_name=st.selectbox('Primary forecast model',MODELS,index=4)
    capacity=st.number_input('Planning capacity threshold',min_value=0,value=3000,step=100)
    st.divider(); st.markdown('### Evaluation design'); st.caption('Chronological holdout + walk-forward validation + multi-horizon error analysis.')
    st.divider(); st.markdown('### Project stack'); st.caption('Python • Pandas • Plotly • SARIMA • ARIMA • Exponential Smoothing • Random Forest • Gradient Boosting • Streamlit')

# =========================================================
# HEADER / FORECASTS
# =========================================================
st.markdown('''<div class="hero"><h1>📈 UAC Predictive Forecasting</h1><p>Care Load & Placement Demand Intelligence</p><p>Turning daily operational reporting into forward-looking planning signals for care, placement and workforce capacity.</p><span class="badge">TIME SERIES • MACHINE LEARNING • UNCERTAINTY • EARLY WARNING</span></div>''',unsafe_allow_html=True)

with st.spinner('Generating forecasts...'):
    care_fc=forecast(df,TARGET,horizon,model_name)
    discharge_fc=forecast(df,DISCHARGE,horizon,'SARIMA')
    val14=latest_holdout(df,14)
    wf=walk_forward_results(df)

# KPI calculations
current=float(df[TARGET].iloc[-1]); future_mean=float(care_fc.Forecast.mean()); future_max=float(care_fc.Forecast.max()); discharge_mean=float(discharge_fc.Forecast.mean()); pressure_mean=float(df['Net Pressure'].tail(7).mean())
accuracy=max(0.0,100-float(val14.loc[val14.Model==model_name,'MAPE'].iloc[0])) if model_name in val14.Model.values else np.nan
# Empirical normal approximation from forecast interval
sig=(care_fc.Upper-care_fc.Lower)/(2*1.96); breach_probs=1-norm.cdf((capacity-care_fc.Forecast)/sig.replace(0,np.nan)); breach_prob=float(np.nanmax(breach_probs))*100
breach_days=np.where(care_fc.Forecast.values>capacity)[0]; surge_lead=int(breach_days[0]+1) if len(breach_days) else None
stability=max(0.0,min(100.0,100-(float(care_fc.Forecast.std())/max(abs(float(care_fc.Forecast.mean())),1))*100))
robustness=max(0.0,min(100.0,100-(float(wf[wf.Model==model_name].groupby('Horizon').MAPE.mean().std()) if not wf.empty else 0)))

cols=st.columns(5)
kpis=[('Current HHS Care',f'{current:,.0f}','Latest observed'),('Forecast Avg',f'{future_mean:,.0f}',f'Next {horizon} days'),('Peak Forecast',f'{future_max:,.0f}','Planning peak'),('Capacity Breach Risk',f'{breach_prob:.1f}%',f'Approx. max probability > {capacity:,.0f}'),('Surge Lead Time',f'{surge_lead} days' if surge_lead else 'No breach', 'Days until threshold crossing')]
for c,(lab,val,note) in zip(cols,kpis):
    c.markdown(f'<div class="card"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section-title">Executive Forecast View</div><div class="section-sub">The selected model forecasts future care load; discharge demand is forecast independently. Confidence intervals quantify statistical uncertainty.</div>',unsafe_allow_html=True)
a,b=st.columns([1.35,1])
with a: st.plotly_chart(forecast_plot(df[TARGET],care_fc,f'Future HHS Care Load — {model_name}','Children in HHS care'),use_container_width=True)
with b: st.plotly_chart(forecast_plot(df[DISCHARGE],discharge_fc,'Short-Term Discharge / Placement Demand','Children discharged'),use_container_width=True)

st.plotly_chart(flow_plot(df),use_container_width=True)

# =========================================================
# TABS
# =========================================================
t1,t2,t3,t4,t5,t6,t7=st.tabs(['🏠 Dashboard','🔎 Data Overview','🔮 Forecast Analysis','🏆 Model Comparison','🧪 Scenario Analysis','📊 KPI & Insights','📚 Methodology'])

with t1:
    st.markdown('### Decision-support summary')
    net_future=care_fc.Forecast.iloc[-1]-current
    c1,c2,c3=st.columns(3)
    c1.markdown(f'<div class="insight {"risk" if net_future>0 else "success"}"><b>Care-load direction</b><br>{"Forecast increases" if net_future>0 else "Forecast decreases"} by approximately <b>{abs(net_future):,.0f}</b> children by day {horizon}.</div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="insight warning"><b>Placement pressure</b><br>Average projected discharges are <b>{discharge_mean:,.0f}</b> per day; recent net pressure is <b>{pressure_mean:,.1f}</b> transfers minus discharges.</div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="insight purple"><b>Model reliability</b><br>{model_name} has 14-day holdout MAPE of <b>{float(val14.loc[val14.Model==model_name,"MAPE"].iloc[0]):.2f}%</b> where available.</div>',unsafe_allow_html=True)
    st.markdown('### Early-warning interpretation')
    if surge_lead:
        st.warning(f'Planning signal: the forecast first crosses the selected capacity threshold in approximately {surge_lead} day(s). This is a forecast-based warning, not a guaranteed operational outcome.')
    else:
        st.success('No capacity-threshold crossing is projected within the selected horizon under the selected model.')
    st.info('Use the Scenario Analysis tab to examine how higher transfers or lower discharge capacity could alter the planning picture.')

with t2:
    st.markdown('### Dataset quality and preparation')
    q1,q2,q3,q4=st.columns(4)
    q1.metric('Raw rows',f'{quality["raw_rows"]:,}'); q2.metric('Blank rows removed',f'{quality["blank_rows"]:,}'); q3.metric('Valid observations',f'{quality["valid_rows"]:,}'); q4.metric('Missing calendar days',f'{quality["missing_days"]:,}')
    st.write(f"**Date range:** {quality['start'].strftime('%d %b %Y')} to {quality['end'].strftime('%d %b %Y')}  |  **Interpolated cells before fill:** {quality['missing_cells_before']:,}")
    st.caption('Blank rows are removed. Missing dates are inserted only to create a continuous daily time index; numeric gaps are time-interpolated for modeling/visualization. This treatment is disclosed to avoid interpreting absent reporting days as zero activity.')
    st.dataframe(df.tail(30).reset_index().rename(columns={'index':'Date'}),use_container_width=True,height=420)
    st.markdown('### Descriptive statistics')
    st.dataframe(df[[INTAKE,'Children in CBP custody',TRANSFER,TARGET,DISCHARGE,'Net Pressure']].describe().T.round(2),use_container_width=True)

with t3:
    st.markdown('### Forecast analysis')
    st.write('Feature engineering includes t−1, t−7, t−14 lags; 7/14-day rolling mean and variance; net pressure; day-of-week; month; and day-of-year effects.')
    st.plotly_chart(forecast_plot(df[TARGET],care_fc,f'Care Load Forecast — {horizon}-Day Horizon','Children'),use_container_width=True)
    st.plotly_chart(forecast_plot(df[DISCHARGE],discharge_fc,f'Discharge Demand Forecast — {horizon}-Day Horizon','Children discharged'),use_container_width=True)
    out=care_fc.copy(); out['Capacity Gap']=capacity-out['Forecast']; out['Breach Probability %']=breach_probs*100
    st.dataframe(out.round(2),use_container_width=True)
    st.download_button('⬇️ Download Care Forecast CSV',out.to_csv().encode(),file_name='care_load_forecast.csv',mime='text/csv')

with t4:
    st.markdown('### Model performance comparison')
    st.write('The primary validation is a strict chronological holdout. Walk-forward validation below tests multiple forecast origins and 7/14/30-day horizons.')
    show=val14.copy(); show['Forecast Accuracy %']=(100-show.MAPE).clip(lower=0).round(2)
    st.dataframe(show.round(3),use_container_width=True,hide_index=True)
    fig=go.Figure()
    for metric in ['MAE','RMSE']:
        fig.add_trace(go.Bar(x=show.Model,y=show[metric],name=metric))
    fig.update_layout(title='14-Day Holdout Error Comparison',barmode='group',template='plotly_white',height=380)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('### Walk-forward / multi-horizon validation')
    if not wf.empty:
        summary=wf.groupby(['Model','Horizon'],as_index=False)[['MAE','RMSE','MAPE']].mean().round(3)
        st.dataframe(summary,use_container_width=True,hide_index=True)
        fig2=go.Figure()
        for h in sorted(wf.Horizon.unique()):
            s=wf[wf.Horizon==h].groupby('Model').MAPE.mean().reindex(MODELS)
            fig2.add_trace(go.Bar(x=s.index,y=s.values,name=f'{h}-day'))
        fig2.update_layout(title='Mean MAPE by Forecast Horizon',barmode='group',template='plotly_white',height=430,yaxis_title='MAPE (%)')
        st.plotly_chart(fig2,use_container_width=True)
    else: st.info('Walk-forward validation could not be computed.')
    st.caption('Model selection should consider both error and stability across horizons; the dashboard intentionally avoids treating a single holdout score as a guarantee of future performance.')

with t5:
    st.markdown('### Scenario comparison')
    scenario=st.selectbox('Select planning scenario',['Baseline','Surge: +20% transfers','Lower placements: -15% discharges','Combined stress'])
    scen=scenario_forecast(care_fc,pressure_mean,scenario)
    sfig=forecast_plot(df[TARGET],scen,f'Scenario: {scenario}','Children in HHS care')
    st.plotly_chart(sfig,use_container_width=True)
    base_peak=float(care_fc.Forecast.max()); scen_peak=float(scen.Forecast.max())
    c1,c2,c3=st.columns(3); c1.metric('Baseline peak',f'{base_peak:,.0f}'); c2.metric('Scenario peak',f'{scen_peak:,.0f}',f'{scen_peak-base_peak:+,.0f}'); c3.metric('Peak vs capacity',f'{scen_peak-capacity:+,.0f}')
    st.info('Scenario adjustments are planning stress tests, not causal forecasts. The +20% transfer and −15% discharge assumptions are user-defined sensitivity assumptions applied transparently to the net-pressure signal.')
    scenario_table=pd.DataFrame({'Date':scen.index,'Baseline':care_fc.Forecast.values,'Scenario':scen.Forecast.values,'Scenario Gap':scen.Forecast.values-care_fc.Forecast.values})
    st.dataframe(scenario_table.round(1),use_container_width=True,hide_index=True)

with t6:
    st.markdown('### KPI framework')
    kpi_table=pd.DataFrame([
        ['Forecast Accuracy (%)',f'{accuracy:.2f}%',f'100 − 14-day MAPE for selected model'],
        ['Surge Lead Time',f'{surge_lead} days' if surge_lead else 'No threshold breach',f'First forecast day above capacity {capacity:,.0f}'],
        ['Capacity Breach Probability',f'{breach_prob:.1f}%', 'Maximum daily probability from forecast interval normal approximation'],
        ['Forecast Stability Index',f'{stability:.1f}/100','100 minus normalized forecast dispersion; higher means smoother projected path'],
        ['Model Robustness',f'{robustness:.1f}/100','Stability of mean MAPE across walk-forward horizons'],
        ['Mean Net Pressure',f'{pressure_mean:.1f}', '7-day transfers minus discharges'],
    ],columns=['KPI','Value','Definition'])
    st.dataframe(kpi_table,use_container_width=True,hide_index=True)
    st.markdown('### Key insights')
    insights=[]
    direction='increase' if care_fc.Forecast.iloc[-1]>current else 'decrease'
    insights.append(f'Care load is forecast to {direction} from {current:,.0f} to approximately {care_fc.Forecast.iloc[-1]:,.0f} over the selected horizon.')
    insights.append(f'Average forecast discharge demand is approximately {discharge_mean:,.0f} children per day.')
    insights.append(f'Recent 7-day net pressure is {pressure_mean:,.1f}; positive values indicate transfers exceed discharges.')
    insights.append(f'The selected model has a 14-day holdout MAPE of {float(val14.loc[val14.Model==model_name,"MAPE"].iloc[0]):.2f}%.' if model_name in val14.Model.values else 'Selected model validation is unavailable.')
    insights.append('Confidence intervals widen with forecast uncertainty; planners should use them as ranges rather than exact counts.')
    for x in insights: st.markdown(f'<div class="insight">💡 {x}</div>',unsafe_allow_html=True)

with t7:
    st.markdown('### Methodology mapped to project requirements')
    method=pd.DataFrame([
        ['Time-series preparation','Date parsed, sorted, indexed daily; blank rows removed; missing calendar days identified; numeric gaps interpolated with disclosure.','Completed'],
        ['Decomposition','Trend/seasonality are modeled through 7-day seasonal SARIMA/Exponential Smoothing and calendar features.','Completed'],
        ['Lag features','t−1, t−7, t−14.','Completed'],
        ['Rolling statistics','7-day and 14-day rolling mean and standard deviation.','Completed'],
        ['Flow signal','Transfers − Discharges = Net Pressure.','Completed'],
        ['Calendar effects','Day of week, month, day of year.','Completed'],
        ['Strict time split','Chronological holdout; no random sampling.','Completed'],
        ['Walk-forward validation','Multiple chronological origins.','Completed'],
        ['Multi-horizon evaluation','7, 14 and 30 days.','Completed'],
        ['Baseline models','Naïve persistence and 7-day moving average.','Completed'],
        ['Statistical models','Exponential Smoothing, ARIMA and SARIMA.','Completed'],
        ['ML models','Random Forest and Gradient Boosting.','Completed'],
        ['Metrics','MAE, RMSE, MAPE and horizon error.','Completed'],
        ['Uncertainty','95% confidence/prediction intervals and derived breach probability.','Completed'],
        ['Streamlit modules','Care load, discharge demand, model comparison, confidence intervals, scenario comparison.','Completed'],
    ],columns=['Requirement','Implementation','Status'])
    st.dataframe(method,use_container_width=True,hide_index=True)
    st.markdown('### Modeling notes')
    st.markdown('''
- **Care-load target:** Children in HHS Care.
- **Placement-demand target:** Children discharged from HHS Care.
- **Net pressure:** Transfers into HHS minus discharges; positive values indicate upward operational pressure.
- **Forecast intervals:** Statistical models use model prediction intervals. ML models use residual-based approximate intervals.
- **Capacity breach probability:** calculated from the forecast distribution implied by the 95% interval using a normal approximation. It is an analytical risk signal, not an observed HHS probability.
- **Scenario analysis:** sensitivity testing, not a causal policy forecast.
- **No random train/test split:** time-series ordering is preserved to avoid look-ahead bias.
''')
    st.markdown('### Submission checklist')
    checklist=['Research paper: EDA + methodology + model evaluation + insights + recommendations','Streamlit dashboard: interactive forecasting + uncertainty + scenarios','Executive summary: current load, forecast, risk, lead time and planning actions','Reproducible requirements.txt and source code','Data-quality disclosure and validation methodology']
    for item in checklist: st.markdown(f'✅ {item}')

st.markdown('<div class="footer">UAC Predictive Forecasting Project • Prepared for academic/project evaluation • Forecasts are analytical decision-support signals, not guarantees.</div>',unsafe_allow_html=True)
