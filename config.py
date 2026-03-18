import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ENTSO-E
ENTSOE_API_KEY = os.environ.get('ENTSOE_API_KEY', '')
COUNTRY_CODE = 'FR'

# Open-Meteo
WEATHER_LAT = 48.8566
WEATHER_LON = 2.3522

# Data
START_DATE = '2015-01-01'
MAX_ENCODER_LENGTH = 168   # 7 days lookback
MAX_PREDICTION_LENGTH = 24 # predict 24 hours at once

# Aliases
SEQ_LEN = MAX_ENCODER_LENGTH
HORIZON = MAX_PREDICTION_LENGTH

# Walk-Forward CV (hours)
WF_TRAIN_SIZE = 365 * 24 * 3
WF_VAL_SIZE   = 60  * 24
WF_TEST_SIZE  = 90  * 24
WF_STEP_SIZE  = 90  * 24

# TFT
TFT_HIDDEN_SIZE       = 32
TFT_ATTENTION_HEADS   = 4
TFT_DROPOUT           = 0.1
TFT_HIDDEN_CONTINUOUS = 16
TFT_LEARNING_RATE     = 1e-3

# Training
BATCH_SIZE = 128
EPOCHS     = 50
PATIENCE   = 8
SEED       = 42