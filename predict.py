import os
import numpy as np
import pandas as pd
import torch
import joblib
import requests
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from features import (
    prepare_tft_dataframe,
    TIME_VARYING_KNOWN_CATEGORICALS,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS,
    STATIC_CATEGORICALS,
)
from config import (
    SAVE_DIR, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
    ENTSOE_API_KEY, COUNTRY_CODE, WEATHER_LAT, WEATHER_LON,
)

if not ENTSOE_API_KEY:
    raise ValueError(
        'ENTSOE_API_KEY not set. Run: export ENTSOE_API_KEY="your-key-here"'
    )

# How many days of history to fetch (need at least encoder length + buffer)
FETCH_DAYS = 30

print('fetching recent data...')

# Load
client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
end   = datetime.now()
start = end - timedelta(days=FETCH_DAYS)

load = client.query_load(
    COUNTRY_CODE,
    start=pd.Timestamp(start, tz='Europe/Paris'),
    end=pd.Timestamp(end, tz='Europe/Paris'),
)
if isinstance(load, pd.DataFrame):
    load = load.iloc[:, 0]
load_df = load.to_frame(name='load_mw')
load_df.index = load_df.index.tz_convert('UTC')
load_df = load_df.resample('h').mean().dropna()
print(f'  load: {len(load_df)} hours')

# Weather
url = 'https://archive-api.open-meteo.com/v1/archive'
params = {
    'latitude': WEATHER_LAT,
    'longitude': WEATHER_LON,
    'start_date': start.strftime('%Y-%m-%d'),
    'end_date': end.strftime('%Y-%m-%d'),
    'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,cloud_cover',
    'timezone': 'UTC',
}
resp = requests.get(url, params=params, timeout=60)
resp.raise_for_status()
hourly = resp.json()['hourly']
wdf = pd.DataFrame(hourly)
wdf['time'] = pd.to_datetime(wdf['time'])
wdf.set_index('time', inplace=True)
wdf.index = wdf.index.tz_localize('UTC')
print(f'  weather: {len(wdf)} hours')

# Merge
df = load_df.join(wdf, how='inner').dropna()
df['hour']        = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month']       = df.index.month
df['is_weekend']  = (df.index.dayofweek >= 5).astype(float)

french_holidays = set()
for year in [2025, 2026, 2027]:
    french_holidays.update([
        f'{year}-01-01', f'{year}-05-01', f'{year}-05-08',
        f'{year}-07-14', f'{year}-08-15', f'{year}-11-01',
        f'{year}-11-11', f'{year}-12-25',
    ])
df['is_holiday'] = df.index.strftime('%Y-%m-%d').isin(french_holidays).astype(float)

print(f'  merged: {len(df)} hours ({len(df)//24} days)')

# Prepare for TFT
tft_df = prepare_tft_dataframe(df)
print(f'  after feature engineering: {len(tft_df)} rows')

# Load model checkpoint
print('\nloading model...')
checkpoint_dir = f'{SAVE_DIR}/checkpoints'
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')] if os.path.exists(checkpoint_dir) else []

if checkpoint_files:
    # Use the most recent checkpoint
    checkpoint_files.sort()
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
    print(f'  checkpoint: {checkpoint_files[-1]}')
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
else:
    raise FileNotFoundError(
        f'No checkpoint found in {checkpoint_dir}. Run train.py first.'
    )

model.eval()

# Create dataset for prediction
# TFT needs a dataset object even for inference
dataset = TimeSeriesDataSet(
    tft_df,
    time_idx='time_idx',
    target='load_mw',
    group_ids=['group'],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    min_encoder_length=MAX_ENCODER_LENGTH,
    min_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=STATIC_CATEGORICALS,
    time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
    time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
    target_normalizer=GroupNormalizer(groups=['group'], transformation='softplus'),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

dl = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

# Get predictions
print('running inference...\n')
raw_preds = model.predict(dl, mode='raw', return_x=True, trainer_kwargs=dict(accelerator='auto'))

predictions = raw_preds.output
if isinstance(predictions, dict):
    pred_tensor = predictions['prediction']
else:
    pred_tensor = predictions

# QuantileLoss default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
pred_np = pred_tensor.cpu().numpy()

# Take the last prediction window (most recent)
last_pred = pred_np[-1]  # shape: (prediction_length, n_quantiles)
q02  = last_pred[:, 0]
q10  = last_pred[:, 1]
q25  = last_pred[:, 2]
q50  = last_pred[:, 3]  # median
q75  = last_pred[:, 4]
q90  = last_pred[:, 5]
q98  = last_pred[:, 6]

# Get actual values for comparison (if available in the dataset)
actuals_list = []
for x, y in dl:
    actuals_list.append(y[0].cpu().numpy())
actuals = np.concatenate(actuals_list, axis=0)
last_actual = actuals[-1]  # shape: (prediction_length,)

# Figure out timestamps for the prediction window
last_timestamp = df.index[-1]
last_known_load = float(df['load_mw'].iloc[-1])
last_temp = float(df['temperature_2m'].iloc[-1])

print(f'Last known data: {last_timestamp.strftime("%Y-%m-%d %H:%M UTC")}')
print(f'Current load: {last_known_load:,.0f} MW | Temperature: {last_temp:.1f} C')
print(f'\nForecast for next {MAX_PREDICTION_LENGTH} hours:')
print(f'{"Hour":<22} | {"Median":>10} | {"10th-90th":>18} | {"Actual":>10} | {"Error":>10}')
print('-' * 82)

errors = []
for h in range(MAX_PREDICTION_LENGTH):
    target_time = last_timestamp + timedelta(hours=h + 1)
    time_str = target_time.strftime('%Y-%m-%d %H:%M')

    median = q50[h]
    low    = q10[h]
    high   = q90[h]

    actual_str = ''
    error_str  = ''
    if h < len(last_actual) and last_actual[h] > 0:
        actual_val = last_actual[h]
        error = median - actual_val
        errors.append(abs(error))
        actual_str = f'{actual_val:>10,.0f}'
        error_str  = f'{error:>+10,.0f}'
    else:
        actual_str = f'{"pending":>10}'
        error_str  = f'{"":>10}'

    print(f'{time_str:<22} | {median:>10,.0f} | {low:>8,.0f} - {high:>8,.0f} | {actual_str} | {error_str}')

print('-' * 82)

if errors:
    mae = np.mean(errors)
    print(f'MAE: {mae:,.0f} MW')
    mape_val = np.mean(np.array(errors) / last_actual[:len(errors)]) * 100
    print(f'MAPE: {mape_val:.2f}%')

# Summary
print(f'\nSummary:')
print(f'  Peak forecast: {q50.max():,.0f} MW at +{q50.argmax()+1}h')
print(f'  Trough forecast: {q50.min():,.0f} MW at +{q50.argmin()+1}h')
print(f'  Average forecast: {q50.mean():,.0f} MW')
print(f'  Uncertainty range (90th pctile): {(q90 - q10).mean():,.0f} MW avg width')

print('\ndone!')