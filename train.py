import os
import warnings
import random
import shutil
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightning.pytorch as pl
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
)
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

from features import (
    prepare_tft_dataframe,
    TIME_VARYING_KNOWN_CATEGORICALS,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS,
    STATIC_CATEGORICALS,
)
from config import (
    SAVE_DIR, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
    WF_TRAIN_SIZE, WF_VAL_SIZE, WF_TEST_SIZE, WF_STEP_SIZE,
    TFT_HIDDEN_SIZE, TFT_ATTENTION_HEADS, TFT_DROPOUT,
    TFT_HIDDEN_CONTINUOUS, TFT_LEARNING_RATE,
    BATCH_SIZE, EPOCHS, PATIENCE, SEED,
)
torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore', category=UserWarning)
pl.seed_everything(SEED)


class EpochLogger(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and 'loss' in outputs:
            self.train_losses.append(outputs['loss'].item())
        elif isinstance(outputs, torch.Tensor):
            self.train_losses.append(outputs.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1

        if self.train_losses:
            train_loss = np.mean(self.train_losses)
            self.train_losses.clear()
        else:
            train_loss = float('nan')

        val_loss = trainer.callback_metrics.get('val_loss', float('nan'))
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        lr = trainer.optimizers[0].param_groups[0]['lr']

        # Store history for plotting
        if not np.isnan(train_loss) and not np.isnan(val_loss):
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.lr_history.append(lr)

        best_val = trainer.early_stopping_callback.best_score if hasattr(trainer, 'early_stopping_callback') else None
        is_best = ''
        if best_val is not None and isinstance(best_val, torch.Tensor):
            if abs(val_loss - best_val.item()) < 1e-6:
                is_best = ' *'

        print(f'    ep {epoch:3d} | train_loss: {train_loss:.4f} | '
              f'val_loss: {val_loss:.4f} | lr: {lr:.2e}{is_best}')

print('loading data...')
raw_df = pd.read_parquet(f'{SAVE_DIR}/france_load_weather.parquet')
df = prepare_tft_dataframe(raw_df)
N = len(df)
print(f'total rows: {N} ({N // 24} days)')
print(f'encoder: {MAX_ENCODER_LENGTH}h | decoder: {MAX_PREDICTION_LENGTH}h')
print(f'load range: {df["load_mw"].min():.0f} - {df["load_mw"].max():.0f} MW')


def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_dataset(data, max_encoder=MAX_ENCODER_LENGTH,
                   max_prediction=MAX_PREDICTION_LENGTH):
    return TimeSeriesDataSet(
        data,
        time_idx='time_idx',
        target='load_mw',
        group_ids=['group'],
        max_encoder_length=max_encoder,
        max_prediction_length=max_prediction,
        min_encoder_length=max_encoder,
        min_prediction_length=max_prediction,
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


def train_tft_fold(train_data, val_data, fold_label=''):
    train_cutoff = train_data['time_idx'].max()
    training = create_dataset(train_data)

    # CRITICAL: min_prediction_idx ensures validation windows only predict
    # AFTER the training period, preventing data leakage
    validation = TimeSeriesDataSet.from_dataset(
        training, val_data,
        min_prediction_idx=train_cutoff + 1,
        stop_randomization=True,
    )

    train_dl = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dl   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=TFT_HIDDEN_SIZE,
        attention_head_size=TFT_ATTENTION_HEADS,
        dropout=TFT_DROPOUT,
        hidden_continuous_size=TFT_HIDDEN_CONTINUOUS,
        learning_rate=TFT_LEARNING_RATE,
        loss=QuantileLoss(),
        log_interval=0,
        optimizer='adam',
        reduce_on_plateau_patience=4,
    )

    total_params = sum(p.numel() for p in tft.parameters())
    if fold_label:
        print(f'  TFT params: {total_params:,}')

    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, mode='min', verbose=False
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss', mode='min',
        dirpath=f'{SAVE_DIR}/checkpoints',
        filename=f'tft-fold{fold_label}' + '-{epoch:02d}-{val_loss:.4f}',
    )
    epoch_logger = EpochLogger()

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        enable_model_summary=False,
        callbacks=[early_stop, checkpoint, epoch_logger],
        enable_progress_bar=False,
        log_every_n_steps=50,
        gradient_clip_val=0.1,
    )

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_val = early_stop.best_score
    stopped = trainer.current_epoch + 1
    best_ep = stopped - early_stop.wait_count
    if isinstance(best_val, torch.Tensor):
        best_val = best_val.item()
    print(f'  training complete: stopped at epoch {stopped}, '
          f'best val_loss: {best_val:.4f} (epoch {best_ep})')

    # Load best checkpoint
    best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
    return best_model, training, epoch_logger, checkpoint.best_model_path


def predict_tft(model, dataset, data, min_prediction_idx=None):
    kwargs = dict(stop_randomization=True)
    if min_prediction_idx is not None:
        kwargs['min_prediction_idx'] = min_prediction_idx

    ds = TimeSeriesDataSet.from_dataset(dataset, data, **kwargs)
    dl = ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    # Get predictions — handle different pytorch-forecasting return formats
    raw_preds = model.predict(dl, mode='raw', return_x=True, trainer_kwargs=dict(accelerator='auto'))

    # QuantileLoss default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    output = raw_preds.output
    if isinstance(output, dict):
        pred_tensor = output['prediction']
    elif isinstance(output, (tuple, list)):
        pred_tensor = output[0]
    else:
        pred_tensor = output

    # pred_tensor shape: (n_samples, prediction_length, n_quantiles)
    median_idx = 3  # 0.5 quantile
    pred_np = pred_tensor.cpu().numpy()
    median_preds = pred_np[:, :, median_idx]

    # Get actuals
    actuals_list = []
    for x, y in dl:
        if isinstance(y, (tuple, list)):
            actuals_list.append(y[0].cpu().numpy())
        else:
            actuals_list.append(y.cpu().numpy())
    actuals = np.concatenate(actuals_list, axis=0)

    return actuals, median_preds, pred_np


# XGBoost baseline
def train_xgb_fold(df_train, df_val, df_test):
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor

    feature_cols = (TIME_VARYING_KNOWN_REALS +
                    [c for c in TIME_VARYING_UNKNOWN_REALS if c != 'load_mw'])

    def make_flat_multistep(data):
        X = data[feature_cols].values
        y_list = []
        for i in range(1, MAX_PREDICTION_LENGTH + 1):
            y_list.append(data['load_mw'].shift(-i).values)
        y = np.column_stack(y_list)
        mask = ~np.isnan(y).any(axis=1)
        return X[mask].astype(np.float32), y[mask].astype(np.float32)

    X_tr, y_tr     = make_flat_multistep(df_train)
    X_val, y_val   = make_flat_multistep(df_val)
    X_test, y_test = make_flat_multistep(df_test)

    base_xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1,
    )
    xgb_multi = MultiOutputRegressor(base_xgb)
    xgb_multi.fit(X_tr, y_tr)

    pred = xgb_multi.predict(X_test)
    mae = mean_absolute_error(y_test.flatten(), pred.flatten())
    return xgb_multi, mae, y_test, pred


# Walk-forward folds
def generate_folds():
    fold_idx = 0
    while True:
        train_start = fold_idx * WF_STEP_SIZE
        train_end   = train_start + WF_TRAIN_SIZE
        val_end     = train_end + WF_VAL_SIZE
        test_end    = val_end + WF_TEST_SIZE

        if test_end + MAX_PREDICTION_LENGTH >= N:
            break
        yield fold_idx, train_start, train_end, val_end, test_end
        fold_idx += 1


print('\nWALK-FORWARD CROSS VALIDATION')
print(f'Train: {WF_TRAIN_SIZE // 24}d | Val: {WF_VAL_SIZE // 24}d | '
      f'Test: {WF_TEST_SIZE // 24}d | Step: {WF_STEP_SIZE // 24}d')

folds = list(generate_folds())
print(f'{len(folds)} folds\n')

has_xgb = True
try:
    from xgboost import XGBRegressor
except ImportError:
    has_xgb = False

oos_actuals_all  = []
oos_medians_all  = []
oos_quantiles_all = []
oos_actuals_2d   = []
oos_medians_2d   = []
oos_timestamps   = []
fold_loss_histories = []
fold_metrics     = []
best_model = None
best_dataset = None
best_checkpoint_path = None

for fold_idx, train_start, train_end, val_end, test_end in folds:
    fold_df = df.iloc[train_start:test_end].copy()
    fold_df['time_idx'] = np.arange(len(fold_df))

    train_data = fold_df.iloc[:train_end - train_start]
    val_data   = fold_df.iloc[:val_end - train_start]
    test_data  = fold_df.iloc[:test_end - train_start]

    t = lambda i: str(df.iloc[i]['timestamp'])[:10] if i < N else '?'
    print(f'Fold {fold_idx + 1}/{len(folds)}')
    print(f'  Train: {t(train_start)} to {t(train_end - 1)}')
    print(f'  Val:   {t(train_end)} to {t(val_end - 1)}')
    print(f'  Test:  {t(val_end)} to {t(test_end - 1)}')
    print(f'  Rows:  train={train_end - train_start} val={val_end - train_end} test={test_end - val_end}')

    # TFT
    tft_model, tft_dataset, fold_logger, best_ckpt_path = train_tft_fold(
        train_data, val_data, fold_label=str(fold_idx + 1)
    )

    fold_loss_histories.append({
        'fold': fold_idx + 1,
        'train_loss': fold_logger.train_loss_history,
        'val_loss': fold_logger.val_loss_history,
        'lr': fold_logger.lr_history,
    })

    # Predict on test data with proper boundary
    # min_prediction_idx ensures we only predict AFTER the val period
    test_min_idx = val_end - train_start
    actuals, medians, quantiles = predict_tft(
        tft_model, tft_dataset, test_data,
        min_prediction_idx=test_min_idx
    )

    # Flatten multi-horizon to per-target for metrics
    # actuals shape: (n_windows, 24), medians shape: (n_windows, 24)
    n_windows = actuals.shape[0]
    act_flat  = actuals.flatten()
    med_flat  = medians.flatten()

    # Build proper timestamps for each forecast target
    # Each window starts at a different origin; each target is origin + (h+1) hours
    test_slice = fold_df.iloc[val_end - train_start : test_end - train_start]
    if 'timestamp' in test_slice.columns:
        test_timestamps = pd.to_datetime(test_slice['timestamp'].values)
    else:
        test_timestamps = None

    fold_target_timestamps = []
    if test_timestamps is not None and len(test_timestamps) > 0:
        # The dataset creates windows starting from min_prediction_idx
        # Each window's origin is a row in the test period
        # We approximate by stepping through windows
        window_origins = test_timestamps[:n_windows]
        for w_idx in range(n_windows):
            if w_idx < len(window_origins):
                origin = window_origins[w_idx]
                for h in range(MAX_PREDICTION_LENGTH):
                    fold_target_timestamps.append(origin + pd.Timedelta(hours=h))

    # Core metrics
    tft_mae  = mean_absolute_error(act_flat, med_flat)
    tft_rmse = np.sqrt(mean_squared_error(act_flat, med_flat))
    tft_mape = mape(act_flat, med_flat)

    ss_res = np.sum((act_flat - med_flat) ** 2)
    ss_tot = np.sum((act_flat - act_flat.mean()) ** 2)
    tft_r2 = 1 - (ss_res / (ss_tot + 1e-10))

    tft_medae = np.median(np.abs(act_flat - med_flat))

    q10_flat = quantiles[:, :, 1].flatten()
    q90_flat = quantiles[:, :, 5].flatten()
    coverage = np.mean((act_flat >= q10_flat) & (act_flat <= q90_flat)) * 100

    bias = np.mean(med_flat - act_flat)
    max_ae = np.max(np.abs(act_flat - med_flat))

    print(f'  TFT OOS metrics:')
    print(f'    MAE:      {tft_mae:>8,.0f} MW')
    print(f'    RMSE:     {tft_rmse:>8,.0f} MW')
    print(f'    MAPE:     {tft_mape:>8.2f}%')
    print(f'    R2:       {tft_r2:>8.4f}')
    print(f'    MedAE:    {tft_medae:>8,.0f} MW')
    print(f'    Coverage: {coverage:>8.1f}% (actuals within 10th-90th quantile, target: 80%)')
    print(f'    Bias:     {bias:>+8,.0f} MW ({"over" if bias > 0 else "under"}predicting)')
    print(f'    Max AE:   {max_ae:>8,.0f} MW (worst single prediction)')

    # Peak vs off-peak using proper timestamps
    if len(fold_target_timestamps) == len(act_flat):
        ts_idx = pd.DatetimeIndex(fold_target_timestamps)
        hour_vals = ts_idx.hour
        peak_mask = ((hour_vals >= 7) & (hour_vals <= 9)) | ((hour_vals >= 18) & (hour_vals <= 20))
        offpeak_mask = ~peak_mask
        if peak_mask.sum() > 0 and offpeak_mask.sum() > 0:
            peak_mae = mean_absolute_error(act_flat[peak_mask], med_flat[peak_mask])
            offpeak_mae = mean_absolute_error(act_flat[offpeak_mask], med_flat[offpeak_mask])
            print(f'    Peak MAE: {peak_mae:>8,.0f} MW (7-9h + 18-20h)')
            print(f'    Off-peak: {offpeak_mae:>8,.0f} MW (all other hours)')

    # XGBoost baseline
    if has_xgb:
        df_tr  = fold_df.iloc[:train_end - train_start]
        df_vl  = fold_df.iloc[train_end - train_start:val_end - train_start]
        df_ts  = fold_df.iloc[val_end - train_start:test_end - train_start]
        _, xgb_mae, _, _ = train_xgb_fold(df_tr, df_vl, df_ts)
        print(f'  XGB OOS: MAE={xgb_mae:.0f} MW (24h multi-output)')
    else:
        xgb_mae = float('nan')

    # Collect OOS data
    oos_actuals_all.append(act_flat)
    oos_medians_all.append(med_flat)
    oos_quantiles_all.append(quantiles)
    oos_actuals_2d.append(actuals)
    oos_medians_2d.append(medians)
    oos_timestamps.extend(fold_target_timestamps)

    # Log per-fold metrics for plotting
    fold_metrics.append({
        'fold': fold_idx + 1,
        'mae': tft_mae,
        'rmse': tft_rmse,
        'mape': tft_mape,
        'r2': tft_r2,
        'medae': tft_medae,
        'coverage': coverage,
        'bias': bias,
        'max_ae': max_ae,
        'xgb_mae': xgb_mae,
        'n_targets': len(act_flat),
        'n_windows': n_windows,
    })

    best_model = tft_model
    best_dataset = tft_dataset
    best_checkpoint_path = best_ckpt_path
    print()

# Aggregate
print('AGGREGATED OOS RESULTS')
all_act = np.concatenate(oos_actuals_all)
all_med = np.concatenate(oos_medians_all)
all_q   = np.concatenate(oos_quantiles_all)

n_total_targets = len(all_act)
n_unique_windows = sum(m['n_windows'] for m in fold_metrics)

total_mae  = mean_absolute_error(all_act, all_med)
total_rmse = np.sqrt(mean_squared_error(all_act, all_med))
total_mape = mape(all_act, all_med)

ss_res = np.sum((all_act - all_med) ** 2)
ss_tot = np.sum((all_act - all_act.mean()) ** 2)
total_r2 = 1 - (ss_res / (ss_tot + 1e-10))

total_medae = np.median(np.abs(all_act - all_med))

q10_all = all_q[:, :, 1].flatten()[:n_total_targets]
q90_all = all_q[:, :, 5].flatten()[:n_total_targets]
total_coverage = np.mean((all_act >= q10_all) & (all_act <= q90_all)) * 100

total_bias = np.mean(all_med - all_act)
total_max_ae = np.max(np.abs(all_act - all_med))
rmse_mae_ratio = total_rmse / (total_mae + 1e-10)

print(f'Total OOS forecast targets: {n_total_targets:,} ({n_unique_windows:,} prediction windows x {MAX_PREDICTION_LENGTH}h)')
print(f'')
print(f'  MAE:      {total_mae:>8,.0f} MW')
print(f'  RMSE:     {total_rmse:>8,.0f} MW')
print(f'  MAPE:     {total_mape:>8.2f}%')
print(f'  R2:       {total_r2:>8.4f}')
print(f'  MedAE:    {total_medae:>8,.0f} MW')
print(f'  Coverage: {total_coverage:>8.1f}% (10th-90th quantile, target: 80%)')
print(f'  Bias:     {total_bias:>+8,.0f} MW ({"over" if total_bias > 0 else "under"}predicting on average)')
print(f'  Max AE:   {total_max_ae:>8,.0f} MW (worst single prediction)')
print(f'  RMSE/MAE: {rmse_mae_ratio:>8.2f} (closer to 1.0 = uniform errors, >>1 = outlier-driven)')
print(f'')
print(f'  Mean load:  {all_act.mean():>10,.0f} MW')
print(f'  MAE as % of mean load: {total_mae / all_act.mean() * 100:.2f}%')

# Per-horizon breakdown
print(f'\n  Per-horizon MAE (hour 1 = next hour, hour {MAX_PREDICTION_LENGTH} = day ahead):')
all_act_2d = np.concatenate(oos_actuals_2d, axis=0)
all_med_2d = np.concatenate(oos_medians_2d, axis=0)
horizon_maes = []
horizon_mapes = []

for h in range(MAX_PREDICTION_LENGTH):
    if h < all_act_2d.shape[1]:
        h_mae = mean_absolute_error(all_act_2d[:, h], all_med_2d[:, h])
        h_mape_val = mape(all_act_2d[:, h], all_med_2d[:, h])
        horizon_maes.append(h_mae)
        horizon_mapes.append(h_mape_val)
        if h in [0, 5, 11, 17, 23]:
            print(f'    Hour {h+1:>2d}: MAE={h_mae:>6,.0f} MW | MAPE={h_mape_val:.2f}%')

# Timestamp-based breakdowns
all_ts = oos_timestamps[:n_total_targets]
has_timestamps = len(all_ts) == n_total_targets

if has_timestamps:
    ts_arr = pd.DatetimeIndex(all_ts)
    hour_vals = ts_arr.hour
    month_vals = ts_arr.month
    dow_vals = ts_arr.dayofweek

    # Peak vs off-peak
    peak_mask = ((hour_vals >= 7) & (hour_vals <= 9)) | ((hour_vals >= 18) & (hour_vals <= 20))
    offpeak_mask = ~peak_mask
    if peak_mask.sum() > 0 and offpeak_mask.sum() > 0:
        peak_mae_val = mean_absolute_error(all_act[peak_mask], all_med[peak_mask])
        peak_mape_val = mape(all_act[peak_mask], all_med[peak_mask])
        offpeak_mae_val = mean_absolute_error(all_act[offpeak_mask], all_med[offpeak_mask])
        offpeak_mape_val = mape(all_act[offpeak_mask], all_med[offpeak_mask])
        print(f'\n  Peak hours (7-9h + 18-20h):')
        print(f'    MAE:  {peak_mae_val:>6,.0f} MW | MAPE: {peak_mape_val:.2f}%')
        print(f'  Off-peak hours:')
        print(f'    MAE:  {offpeak_mae_val:>6,.0f} MW | MAPE: {offpeak_mape_val:.2f}%')

    # Seasonal breakdown
    winter_mask = (month_vals >= 11) | (month_vals <= 2)
    summer_mask = (month_vals >= 5) & (month_vals <= 8)
    spring_fall_mask = ~winter_mask & ~summer_mask

    print(f'\n  Seasonal breakdown:')
    for name, mask in [('Winter (Nov-Feb)', winter_mask),
                       ('Summer (May-Aug)', summer_mask),
                       ('Spring/Fall', spring_fall_mask)]:
        if mask.sum() > 0:
            s_mae = mean_absolute_error(all_act[mask], all_med[mask])
            s_mape_val = mape(all_act[mask], all_med[mask])
            print(f'    {name:<20s}: MAE={s_mae:>6,.0f} MW | MAPE={s_mape_val:.2f}%')

    # Weekday vs weekend
    weekday_mask = dow_vals < 5
    weekend_mask = dow_vals >= 5
    if weekday_mask.sum() > 0 and weekend_mask.sum() > 0:
        wd_mae = mean_absolute_error(all_act[weekday_mask], all_med[weekday_mask])
        we_mae = mean_absolute_error(all_act[weekend_mask], all_med[weekend_mask])
        wd_mape_val = mape(all_act[weekday_mask], all_med[weekday_mask])
        we_mape_val = mape(all_act[weekend_mask], all_med[weekend_mask])
        print(f'\n  Weekday vs Weekend:')
        print(f'    Weekday:  MAE={wd_mae:>6,.0f} MW | MAPE={wd_mape_val:.2f}%')
        print(f'    Weekend:  MAE={we_mae:>6,.0f} MW | MAPE={we_mape_val:.2f}%')
else:
    print('\n  (timestamp breakdown skipped: timestamp count mismatch)')

# Per-fold metrics summary table
print(f'\n  Per-fold summary:')
print(f'    {"Fold":>4s} | {"MAE":>8s} | {"MAPE":>7s} | {"R2":>7s} | {"Bias":>8s} | {"Max AE":>8s} | {"XGB MAE":>8s}')
for m in fold_metrics:
    xgb_str = f'{m["xgb_mae"]:>8,.0f}' if not np.isnan(m['xgb_mae']) else '     N/A'
    print(f'    {m["fold"]:>4d} | {m["mae"]:>8,.0f} | {m["mape"]:>6.2f}% | {m["r2"]:>7.4f} | '
          f'{m["bias"]:>+8,.0f} | {m["max_ae"]:>8,.0f} | {xgb_str}')

# Save production model
print('\nSaving production model (last fold)...')
os.makedirs(f'{SAVE_DIR}/checkpoints', exist_ok=True)
prod_ckpt_path = f'{SAVE_DIR}/checkpoints/tft_production.ckpt'
shutil.copy2(best_checkpoint_path, prod_ckpt_path)
print(f'saved: {prod_ckpt_path}')

# Save interpretability data
print('Extracting variable importance...')
try:
    interp_dl = best_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    raw_output = best_model.predict(
        interp_dl, mode='raw', return_x=True,
        trainer_kwargs=dict(accelerator='auto'),
    )
    output = raw_output.output
    if isinstance(output, (tuple, list)):
        output = output[0]
    interpretation = best_model.interpret_output(output, reduction='mean')
    if hasattr(interpretation, 'keys'):
        interp_dict = {}
        for key in interpretation:
            val = interpretation[key]
            if isinstance(val, torch.Tensor):
                interp_dict[key] = val.cpu().numpy()
            else:
                interp_dict[key] = val
        joblib.dump(interp_dict, f'{SAVE_DIR}/tft_interpretation.pkl')
        print('saved: tft_interpretation.pkl')
except Exception as e:
    print(f'could not extract interpretation: {e}')

# Save OOS results
np.save(f'{SAVE_DIR}/oos_actual.npy', all_act)
np.save(f'{SAVE_DIR}/oos_predicted.npy', all_med)
np.save(f'{SAVE_DIR}/oos_quantiles.npy', all_q)
joblib.dump(fold_loss_histories, f'{SAVE_DIR}/fold_loss_histories.pkl')
joblib.dump(fold_metrics, f'{SAVE_DIR}/fold_metrics.pkl')

# Plot (3x3 grid)
print('\nGenerating plots...')
fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle(f'TFT Walk-Forward: France Load Forecasting ({MAX_PREDICTION_LENGTH}h ahead)', fontsize=16)

# 1. Actual vs predicted scatter
axes[0, 0].scatter(all_act, all_med, alpha=0.03, s=2, color='steelblue')
lims = [min(all_act.min(), all_med.min()), max(all_act.max(), all_med.max())]
axes[0, 0].plot(lims, lims, 'r--', alpha=0.5)
axes[0, 0].set_xlabel('Actual (MW)')
axes[0, 0].set_ylabel('Predicted (MW)')
axes[0, 0].set_title(f'Actual vs Predicted (MAE={total_mae:,.0f} MW)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Error distribution
errors = all_med - all_act
axes[0, 1].hist(errors, bins=100, color='mediumseagreen', alpha=0.7)
axes[0, 1].axvline(x=0, color='red', linestyle='--')
axes[0, 1].set_xlabel('Error (MW)')
axes[0, 1].set_title(f'Error Distribution (bias={total_bias:+,.0f} MW)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Last week overlay
n_show = min(168, len(all_act))
axes[0, 2].plot(all_act[-n_show:], label='Actual', color='steelblue', linewidth=1.5)
axes[0, 2].plot(all_med[-n_show:], label='TFT', color='tomato', linewidth=1.5, alpha=0.8)
axes[0, 2].set_xlabel('Forecast target')
axes[0, 2].set_ylabel('Load (MW)')
axes[0, 2].set_title('Last Week: Actual vs Predicted')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Val loss curves per fold
colors_fold = plt.cm.viridis(np.linspace(0, 1, len(fold_loss_histories)))
for i, hist in enumerate(fold_loss_histories):
    if hist['val_loss']:
        ep_range = range(1, len(hist['val_loss']) + 1)
        axes[1, 0].plot(ep_range, hist['val_loss'], color=colors_fold[i], alpha=0.5, linewidth=1)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Val Loss')
axes[1, 0].set_title('Validation Loss per Fold')
axes[1, 0].grid(True, alpha=0.3)

# 5. MAE per fold
fold_nums = [m['fold'] for m in fold_metrics]
fold_mae_vals = [m['mae'] for m in fold_metrics]
fold_xgb_vals = [m['xgb_mae'] for m in fold_metrics]
bar_x = np.arange(len(fold_nums))
axes[1, 1].bar(bar_x - 0.15, fold_mae_vals, width=0.3, color='steelblue', label='TFT')
if not all(np.isnan(v) for v in fold_xgb_vals):
    axes[1, 1].bar(bar_x + 0.15, fold_xgb_vals, width=0.3, color='darkorange', label='XGBoost')
axes[1, 1].set_xticks(bar_x[::max(1, len(bar_x)//10)])
axes[1, 1].set_xticklabels([str(f) for f in fold_nums[::max(1, len(fold_nums)//10)]])
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('MAE (MW)')
axes[1, 1].set_title('MAE per Fold: TFT vs XGBoost')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. MAPE per fold
fold_mape_vals = [m['mape'] for m in fold_metrics]
axes[1, 2].plot(fold_nums, fold_mape_vals, 'o-', color='darkorange', linewidth=1.5, markersize=4)
axes[1, 2].axhline(y=np.mean(fold_mape_vals), color='gray', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(fold_mape_vals):.2f}%')
axes[1, 2].set_xlabel('Fold')
axes[1, 2].set_ylabel('MAPE (%)')
axes[1, 2].set_title('MAPE per Fold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# 7. Per-horizon MAE
if horizon_maes:
    axes[2, 0].bar(range(1, len(horizon_maes) + 1), horizon_maes, color='orchid', alpha=0.8)
    axes[2, 0].set_xlabel('Hours ahead')
    axes[2, 0].set_ylabel('MAE (MW)')
    axes[2, 0].set_title('MAE by Forecast Horizon')
    axes[2, 0].grid(True, alpha=0.3)

# 8. R2 + Coverage per fold
ax8 = axes[2, 1]
fold_r2_vals = [m['r2'] for m in fold_metrics]
fold_cov_vals = [m['coverage'] for m in fold_metrics]
ax8.plot(fold_nums, fold_r2_vals, 'o-', color='steelblue', linewidth=1.5, markersize=4, label='R2')
ax8_twin = ax8.twinx()
ax8_twin.plot(fold_nums, fold_cov_vals, 's-', color='mediumseagreen', linewidth=1.5, markersize=4, label='Coverage %')
ax8_twin.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
ax8.set_xlabel('Fold')
ax8.set_ylabel('R2', color='steelblue')
ax8_twin.set_ylabel('Coverage %', color='mediumseagreen')
ax8.set_title('R2 and Coverage per Fold')
ax8.grid(True, alpha=0.3)
lines1, labels1 = ax8.get_legend_handles_labels()
lines2, labels2 = ax8_twin.get_legend_handles_labels()
ax8.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# 9. Bias per fold
fold_bias_vals = [m['bias'] for m in fold_metrics]
colors_bias = ['tomato' if b > 0 else 'steelblue' for b in fold_bias_vals]
axes[2, 2].bar(fold_nums, fold_bias_vals, color=colors_bias, alpha=0.7)
axes[2, 2].axhline(y=0, color='black', linewidth=0.5)
axes[2, 2].set_xlabel('Fold')
axes[2, 2].set_ylabel('Bias (MW)')
axes[2, 2].set_title('Bias per Fold (red=over, blue=under)')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/training_results.png', dpi=150)
print(f'plot saved to {SAVE_DIR}/training_results.png')
print('done!')