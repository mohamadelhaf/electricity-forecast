import numpy as np
import pandas as pd


def prepare_tft_dataframe(df):
    out = df.copy()

    # Integer time index (required by pytorch-forecasting)
    out = out.sort_index()
    out['time_idx'] = np.arange(len(out))

    # Group identifier (single series: France)
    out['group'] = 'FR'

    # Cyclical time encodings (known in the future)
    out['hour_sin']  = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos']  = np.cos(2 * np.pi * out['hour'] / 24)
    out['dow_sin']   = np.sin(2 * np.pi * out['day_of_week'] / 7)
    out['dow_cos']   = np.cos(2 * np.pi * out['day_of_week'] / 7)
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)

    # Heating/cooling degree (derived from temperature, known if we have weather forecast)
    out['heating_deg'] = np.maximum(0, 18 - out['temperature_2m'])
    out['cooling_deg'] = np.maximum(0, out['temperature_2m'] - 24)

    # Lagged load features (unknown in future, but useful as encoder features)
    out['load_lag_24h']  = out['load_mw'].shift(24)
    out['load_lag_168h'] = out['load_mw'].shift(168)
    out['load_roll_24h'] = out['load_mw'].rolling(24).mean()

    # Convert categoricals to string for pytorch-forecasting
    out['hour_cat']    = out['hour'].astype(str)
    out['dow_cat']     = out['day_of_week'].astype(str)
    out['month_cat']   = out['month'].astype(str)
    out['weekend_cat'] = out['is_weekend'].astype(int).astype(str)
    out['holiday_cat'] = out['is_holiday'].astype(int).astype(str)

    out.dropna(inplace=True)
    out.reset_index(inplace=True)
    out.rename(columns={'index': 'timestamp'}, inplace=True)

    # Recompute time_idx after dropping NaN
    out['time_idx'] = np.arange(len(out))

    return out


# Feature lists for TFT configuration
TIME_VARYING_KNOWN_CATEGORICALS = [
    'hour_cat', 'dow_cat', 'month_cat', 'weekend_cat', 'holiday_cat',
]

TIME_VARYING_KNOWN_REALS = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
    'shortwave_radiation', 'cloud_cover',
    'heating_deg', 'cooling_deg',
]

TIME_VARYING_UNKNOWN_REALS = [
    'load_mw',
    'load_lag_24h', 'load_lag_168h', 'load_roll_24h',
]

STATIC_CATEGORICALS = ['group']