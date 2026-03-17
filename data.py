import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient

from config import (
    SAVE_DIR, ENTSOE_API_KEY, COUNTRY_CODE,
    WEATHER_LAT, WEATHER_LON, START_DATE,
)

if not ENTSOE_API_KEY:
    raise ValueError(
        'ENTSOE_API_KEY not set. Run: export ENTSOE_API_KEY="your-key-here"\n'
        'Get a free key at https://transparency.entsoe.eu/'
    )

end_date = datetime.today().strftime('%Y-%m-%d')

# Fetch ENTSO-E actual load
print(f'fetching France electricity load from ENTSO-E ({START_DATE} to {end_date})...')
client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

start_ts = pd.Timestamp(START_DATE, tz='Europe/Paris')
end_ts   = pd.Timestamp(end_date, tz='Europe/Paris')

# ENTSO-E has a 1-year limit per request, so we chunk
chunks = []
chunk_start = start_ts
while chunk_start < end_ts:
    chunk_end = min(chunk_start + pd.DateOffset(years=1), end_ts)
    print(f'  load: {chunk_start.date()} to {chunk_end.date()}')
    try:
        load = client.query_load(COUNTRY_CODE, start=chunk_start, end=chunk_end)
        if isinstance(load, pd.DataFrame):
            load = load.iloc[:, 0]
        chunks.append(load)
    except Exception as e:
        print(f'  warning: {e}')
    chunk_start = chunk_end

load_series = pd.concat(chunks)
load_series = load_series[~load_series.index.duplicated(keep='first')]
load_series = load_series.sort_index()
load_series.name = 'load_mw'

# Convert to UTC for alignment with weather data
load_df = load_series.to_frame()
load_df.index = load_df.index.tz_convert('UTC')

# Resample to exact hourly (ENTSO-E sometimes has 15-min or 30-min resolution)
load_df = load_df.resample('h').mean()
load_df = load_df.dropna()

print(f'load data: {len(load_df)} hourly rows')
print(f'  range: {load_df.index[0]} to {load_df.index[-1]}')
print(f'  load min: {load_df["load_mw"].min():.0f} MW')
print(f'  load max: {load_df["load_mw"].max():.0f} MW')
print(f'  load mean: {load_df["load_mw"].mean():.0f} MW')

# Fetch Open-Meteo historical weather
print(f'\nfetching weather data from Open-Meteo (Paris: {WEATHER_LAT}, {WEATHER_LON})...')

weather_vars = [
    'temperature_2m',
    'relative_humidity_2m',
    'wind_speed_10m',
    'shortwave_radiation',
    'cloud_cover',
]


# Chunk by year to be safe
weather_chunks = []
year_start = pd.Timestamp(START_DATE)
final_end  = pd.Timestamp(end_date)

while year_start < final_end:
    year_end = min(year_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), final_end)
    print(f'  weather: {year_start.date()} to {year_end.date()}')

    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': WEATHER_LAT,
        'longitude': WEATHER_LON,
        'start_date': year_start.strftime('%Y-%m-%d'),
        'end_date': year_end.strftime('%Y-%m-%d'),
        'hourly': ','.join(weather_vars),
        'timezone': 'UTC',
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data['hourly']
    wdf = pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        **{v: hourly[v] for v in weather_vars}
    })
    wdf.set_index('time', inplace=True)
    weather_chunks.append(wdf)

    year_start = year_end + pd.DateOffset(days=1)

weather_df = pd.concat(weather_chunks)
weather_df = weather_df[~weather_df.index.duplicated(keep='first')]
weather_df = weather_df.sort_index()
weather_df.index = weather_df.index.tz_localize('UTC')

print(f'weather data: {len(weather_df)} hourly rows')

# Merge load + weather on UTC hourly index
print('\nmerging load + weather...')
df = load_df.join(weather_df, how='inner')
df = df.dropna()

# Add time features (these are crucial for load forecasting)
df['hour']        = df.index.hour
df['day_of_week'] = df.index.dayofweek  # 0=Monday
df['month']       = df.index.month
df['is_weekend']  = (df.index.dayofweek >= 5).astype(float)

# French public holidays (approximate, major ones)
french_holidays = set()
for year in range(2015, 2027):
    french_holidays.update([
        f'{year}-01-01',  # New Year
        f'{year}-05-01',  # Labour Day
        f'{year}-05-08',  # Victory in Europe
        f'{year}-07-14',  # Bastille Day
        f'{year}-08-15',  # Assumption
        f'{year}-11-01',  # All Saints
        f'{year}-11-11',  # Armistice
        f'{year}-12-25',  # Christmas
    ])
df['is_holiday'] = df.index.strftime('%Y-%m-%d').isin(french_holidays).astype(float)

print(f'merged rows: {len(df)}')
print(f'date range: {df.index[0]} to {df.index[-1]}')
print(f'columns: {list(df.columns)}')

# Save
df.to_parquet(f'{SAVE_DIR}/france_load_weather.parquet')
print(f'\nsaved to france_load_weather.parquet')
print('done!')