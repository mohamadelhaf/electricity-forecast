import os
import warnings
import random
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

warnings.filterwarnings('ignore', category=UserWarning)
pl.seed_everything(SEED)


class EpochLogger(pl.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.train_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and 'loss' in outputs:
            self.train_losses.append(outputs['loss'].item())
        elif isinstance(outputs, torch.Tensor):
            self.train_losses.append(outputs.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1

        # Train loss (average over all batches this epoch)
        if self.train_losses:
            train_loss = np.mean(self.train_losses)
            self.train_losses.clear()
        else:
            train_loss = float('nan')

        # Val loss from Lightning's logged metrics
        val_loss = trainer.callback_metrics.get('val_loss', float('nan'))
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        # Learning rate
        lr = trainer.optimizers[0].param_groups[0]['lr']

        # Check if this is the best epoch so far
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


def create_dataset(data, train_cutoff_idx, max_encoder=MAX_ENCODER_LENGTH,
                   max_prediction=MAX_PREDICTION_LENGTH, is_train=True):
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

    training = create_dataset(train_data, train_cutoff)
    validation = TimeSeriesDataSet.from_dataset(training, val_data, stop_randomization=True)

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
    return best_model, training


def predict_tft(model, dataset, data):
    ds = TimeSeriesDataSet.from_dataset(dataset, data, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    raw_preds = model.predict(dl, mode='raw', return_x=True, trainer_kwargs=dict(accelerator='auto'))

    # Extract median (quantile index 1 = 50th percentile for default QuantileLoss)
    # QuantileLoss default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    predictions = raw_preds.output
    if isinstance(predictions, dict):
        pred_tensor = predictions['prediction']
    else:
        pred_tensor = predictions

    # pred_tensor shape: (n_samples, prediction_length, n_quantiles)
    median_idx = 3  # 0.5 quantile
    median_preds = pred_tensor[:, :, median_idx].cpu().numpy()

    # Get actuals
    actuals_list = []
    for x, y in dl:
        actuals_list.append(y[0].cpu().numpy())
    actuals = np.concatenate(actuals_list, axis=0)

    return actuals, median_preds, pred_tensor.cpu().numpy()


# XGBoost baseline
def train_xgb_fold(df_train, df_val, df_test):
    from xgboost import XGBRegressor

    feature_cols = (TIME_VARYING_KNOWN_REALS +
                    [c for c in TIME_VARYING_UNKNOWN_REALS if c != 'load_mw'])

    def make_flat(data):
        X = data[feature_cols].values
        # Target: mean load over next 24h (simplified for XGB)
        y = data['load_mw'].rolling(MAX_PREDICTION_LENGTH).mean().shift(-MAX_PREDICTION_LENGTH).values
        mask = ~np.isnan(y)
        return X[mask].astype(np.float32), y[mask].astype(np.float32)

    X_tr, y_tr   = make_flat(df_train)
    X_val, y_val = make_flat(df_val)
    X_test, y_test = make_flat(df_test)

    xgb = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, early_stopping_rounds=30,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    pred = xgb.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return xgb, mae, y_test, pred


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
oos_actuals_2d   = []  # non-flattened for per-horizon breakdown
oos_medians_2d   = []
best_model = None
best_dataset = None

for fold_idx, train_start, train_end, val_end, test_end in folds:
    fold_df = df.iloc[train_start:test_end].copy()
    fold_df['time_idx'] = np.arange(len(fold_df))

    train_data = fold_df.iloc[:train_end - train_start]
    val_data   = fold_df.iloc[:val_end - train_start]    # includes train (TFT needs history)
    test_data  = fold_df.iloc[:test_end - train_start]   # includes train+val

    t = lambda i: str(df.iloc[i]['timestamp'])[:10] if i < N else '?'
    print(f'Fold {fold_idx + 1}/{len(folds)}')
    print(f'  Train: {t(train_start)} to {t(train_end - 1)}')
    print(f'  Val:   {t(train_end)} to {t(val_end - 1)}')
    print(f'  Test:  {t(val_end)} to {t(test_end - 1)}')
    print(f'  Rows:  train={train_end - train_start} val={val_end - train_end} test={test_end - val_end}')

    # TFT
    tft_model, tft_dataset = train_tft_fold(
        train_data, val_data, fold_label=str(fold_idx + 1)
    )

    actuals, medians, quantiles = predict_tft(tft_model, tft_dataset, test_data)

    # Flatten multi-horizon to per-hour for metrics
    act_flat  = actuals.flatten()
    med_flat  = medians.flatten()

    tft_mae  = mean_absolute_error(act_flat, med_flat)
    tft_rmse = np.sqrt(mean_squared_error(act_flat, med_flat))
    tft_mape = mape(act_flat, med_flat)

    # R2 score
    ss_res = np.sum((act_flat - med_flat) ** 2)
    ss_tot = np.sum((act_flat - act_flat.mean()) ** 2)
    tft_r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # Median absolute error (robust to outliers)
    tft_medae = np.median(np.abs(act_flat - med_flat))

    # Quantile coverage: % of actuals within 10th-90th percentile
    q10_flat = quantiles[:, :, 1].flatten()
    q90_flat = quantiles[:, :, 5].flatten()
    coverage = np.mean((act_flat >= q10_flat) & (act_flat <= q90_flat)) * 100

    print(f'  TFT OOS metrics:')
    print(f'    MAE:      {tft_mae:>8,.0f} MW')
    print(f'    RMSE:     {tft_rmse:>8,.0f} MW')
    print(f'    MAPE:     {tft_mape:>8.2f}%')
    print(f'    R2:       {tft_r2:>8.4f}')
    print(f'    MedAE:    {tft_medae:>8,.0f} MW')
    print(f'    Coverage: {coverage:>8.1f}% (actuals within 10th-90th quantile, target: 80%)')

    # XGBoost baseline
    if has_xgb:
        df_tr  = fold_df.iloc[:train_end - train_start]
        df_vl  = fold_df.iloc[train_end - train_start:val_end - train_start]
        df_ts  = fold_df.iloc[val_end - train_start:test_end - train_start]
        _, xgb_mae, _, _ = train_xgb_fold(df_tr, df_vl, df_ts)
        print(f'  XGB OOS: MAE={xgb_mae:.0f} MW (24h-mean baseline)')

    oos_actuals_all.append(act_flat)
    oos_medians_all.append(med_flat)
    oos_quantiles_all.append(quantiles)
    oos_actuals_2d.append(actuals)
    oos_medians_2d.append(medians)

    best_model = tft_model
    best_dataset = tft_dataset
    print()

# Aggregate
print('AGGREGATED OOS RESULTS')
all_act = np.concatenate(oos_actuals_all)
all_med = np.concatenate(oos_medians_all)
all_q   = np.concatenate(oos_quantiles_all)

total_mae  = mean_absolute_error(all_act, all_med)
total_rmse = np.sqrt(mean_squared_error(all_act, all_med))
total_mape = mape(all_act, all_med)

ss_res = np.sum((all_act - all_med) ** 2)
ss_tot = np.sum((all_act - all_act.mean()) ** 2)
total_r2 = 1 - (ss_res / (ss_tot + 1e-10))

total_medae = np.median(np.abs(all_act - all_med))

q10_all = all_q[:, :, 1].flatten()
q90_all = all_q[:, :, 5].flatten()
total_coverage = np.mean((all_act >= q10_all) & (all_act <= q90_all)) * 100

print(f'Total OOS hours: {len(all_act)}')
print(f'')
print(f'  MAE:      {total_mae:>8,.0f} MW')
print(f'  RMSE:     {total_rmse:>8,.0f} MW')
print(f'  MAPE:     {total_mape:>8.2f}%')
print(f'  R2:       {total_r2:>8.4f}')
print(f'  MedAE:    {total_medae:>8,.0f} MW')
print(f'  Coverage: {total_coverage:>8.1f}% (10th-90th quantile, target: 80%)')
print(f'')
print(f'  Mean load:  {all_act.mean():>10,.0f} MW')
print(f'  MAE as % of mean load: {total_mae / all_act.mean() * 100:.2f}%')

# Per-horizon breakdown (hour 1 vs hour 24 accuracy)
print(f'\n  Per-horizon MAE (hour 1 = next hour, hour 24 = day ahead):')
all_act_2d = np.concatenate(oos_actuals_2d, axis=0)  # (n_windows, 24)
all_med_2d = np.concatenate(oos_medians_2d, axis=0)

for h in [0, 5, 11, 17, 23]:
    if h < all_act_2d.shape[1]:
        h_mae = mean_absolute_error(all_act_2d[:, h], all_med_2d[:, h])
        h_mape = mape(all_act_2d[:, h], all_med_2d[:, h])
        print(f'    Hour {h+1:>2d}: MAE={h_mae:>6,.0f} MW | MAPE={h_mape:.2f}%')

# Save production model
print('\nSaving production model (last fold)...')
os.makedirs(f'{SAVE_DIR}/checkpoints', exist_ok=True)
torch.save(best_model.state_dict(), f'{SAVE_DIR}/tft_model.pth')

# Save interpretability data from last fold
print('Extracting variable importance...')
try:
    interpretation = best_model.interpret_output(
        best_model.predict(
            best_dataset.to_dataloader(train=False, batch_size=64, num_workers=0),
            mode='raw', return_x=True, trainer_kwargs=dict(accelerator='auto'),
        ).output,
        reduction='mean',
    )
    # Save variable importance
    if hasattr(interpretation, 'keys'):
        for key in interpretation:
            if isinstance(interpretation[key], torch.Tensor):
                interpretation[key] = interpretation[key].cpu().numpy()
        joblib.dump(dict(interpretation), f'{SAVE_DIR}/tft_interpretation.pkl')
        print('saved: tft_interpretation.pkl')
except Exception as e:
    print(f'could not extract interpretation: {e}')

# Save OOS results
np.save(f'{SAVE_DIR}/oos_actual.npy', all_act)
np.save(f'{SAVE_DIR}/oos_predicted.npy', all_med)
np.save(f'{SAVE_DIR}/oos_quantiles.npy', all_q)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'TFT Walk-Forward: France Load Forecasting ({MAX_PREDICTION_LENGTH}h ahead)', fontsize=14)

axes[0].scatter(all_act, all_med, alpha=0.05, s=2, color='steelblue')
lims = [min(all_act.min(), all_med.min()), max(all_act.max(), all_med.max())]
axes[0].plot(lims, lims, 'r--', alpha=0.5)
axes[0].set_xlabel('Actual (MW)')
axes[0].set_ylabel('Predicted (MW)')
axes[0].set_title(f'Actual vs Predicted (MAE={total_mae:.0f} MW)')
axes[0].grid(True, alpha=0.3)

errors = all_med - all_act
axes[1].hist(errors, bins=100, color='mediumseagreen', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--')
axes[1].set_xlabel('Error (MW)')
axes[1].set_title(f'Error Distribution (MAPE={total_mape:.2f}%)')
axes[1].grid(True, alpha=0.3)

n_show = min(168, len(all_act))
axes[2].plot(all_act[-n_show:], label='Actual', color='steelblue', linewidth=1.5)
axes[2].plot(all_med[-n_show:], label='TFT', color='tomato', linewidth=1.5, alpha=0.8)
axes[2].set_xlabel('Hour')
axes[2].set_ylabel('Load (MW)')
axes[2].set_title('Last Week: Actual vs Predicted')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/training_results.png', dpi=150)
print(f'\nplot saved to {SAVE_DIR}/training_results.png')
print('done!')