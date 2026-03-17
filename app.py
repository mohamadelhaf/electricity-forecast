import sys
import subprocess
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    SAVE_DIR, COUNTRY_CODE,
    MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
    WF_TRAIN_SIZE, WF_VAL_SIZE, WF_TEST_SIZE, WF_STEP_SIZE,
    TFT_HIDDEN_SIZE, TFT_ATTENTION_HEADS, TFT_DROPOUT,
    TFT_LEARNING_RATE, BATCH_SIZE, EPOCHS, PATIENCE,
    WEATHER_LAT, WEATHER_LON, START_DATE,
)

st.set_page_config(page_title='Electricity Load Forecast', layout='wide')


def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))


def file_exists(name):
    return os.path.exists(os.path.join(SAVE_DIR, name))


def run_script(name):
    result = subprocess.run(
        [sys.executable, os.path.join(SAVE_DIR, name)],
        capture_output=True, text=True, cwd=SAVE_DIR,
    )
    return result.stdout + result.stderr


# Sidebar
st.sidebar.title('Electricity Load Forecast')
st.sidebar.caption(f'Country: {COUNTRY_CODE}')
st.sidebar.markdown('---')

page = st.sidebar.radio('Navigate', [
    'Forecast',
    'Performance',
    'Insights',
    'Pipeline',
])

st.sidebar.markdown('---')
st.sidebar.markdown('Model Configuration')
st.sidebar.markdown(f'Architecture: TFT')
st.sidebar.markdown(f'Hidden size: {TFT_HIDDEN_SIZE}')
st.sidebar.markdown(f'Attention heads: {TFT_ATTENTION_HEADS}')
st.sidebar.markdown(f'Dropout: {TFT_DROPOUT}')
st.sidebar.markdown(f'Learning rate: {TFT_LEARNING_RATE}')
st.sidebar.markdown(f'Batch size: {BATCH_SIZE}')

st.sidebar.markdown('---')
st.sidebar.markdown('Data Configuration')
st.sidebar.markdown(f'Lookback: {MAX_ENCODER_LENGTH}h ({MAX_ENCODER_LENGTH // 24} days)')
st.sidebar.markdown(f'Horizon: {MAX_PREDICTION_LENGTH}h')
st.sidebar.markdown(f'Data from: {START_DATE}')
st.sidebar.markdown(f'Weather: Paris ({WEATHER_LAT}, {WEATHER_LON})')

st.sidebar.markdown('---')
st.sidebar.markdown('Walk-Forward CV')
st.sidebar.markdown(f'Train window: {WF_TRAIN_SIZE // 24} days')
st.sidebar.markdown(f'Val window: {WF_VAL_SIZE // 24} days')
st.sidebar.markdown(f'Test window: {WF_TEST_SIZE // 24} days')
st.sidebar.markdown(f'Step: {WF_STEP_SIZE // 24} days')

st.sidebar.markdown('---')
st.sidebar.caption('Data: ENTSO-E + Open-Meteo')


if page == 'Forecast':
    st.title('France Electricity Load Forecast')
    st.markdown(
        f'The model looks at the last {MAX_ENCODER_LENGTH} hours '
        f'({MAX_ENCODER_LENGTH // 24} days) of load and weather data, '
        f'then predicts the next {MAX_PREDICTION_LENGTH} hours of '
        f'electricity demand across France.'
    )

    if not file_exists('oos_actual.npy'):
        st.warning(
            'No predictions found. Go to the Pipeline page and run '
            'Data → Train → Predict first.'
        )
        st.stop()

    oos_actual = np.load(os.path.join(SAVE_DIR, 'oos_actual.npy'))
    oos_pred   = np.load(os.path.join(SAVE_DIR, 'oos_predicted.npy'))

    # Show the most recent 24h prediction window
    n_windows = len(oos_actual) // MAX_PREDICTION_LENGTH
    if n_windows == 0:
        st.error('Not enough data for a full prediction window.')
        st.stop()

    last_window_actual = oos_actual[-MAX_PREDICTION_LENGTH:]
    last_window_pred   = oos_pred[-MAX_PREDICTION_LENGTH:]

    # Load quantiles if available
    has_quantiles = file_exists('oos_quantiles.npy')
    if has_quantiles:
        oos_q = np.load(os.path.join(SAVE_DIR, 'oos_quantiles.npy'))
        # Last window: shape (prediction_length, n_quantiles)
        last_q = oos_q[-1] if oos_q.ndim == 3 else None
    else:
        last_q = None

    current_load = last_window_actual[0]
    peak_pred    = last_window_pred.max()
    trough_pred  = last_window_pred.min()
    avg_pred     = last_window_pred.mean()
    window_mae   = mean_absolute_error(last_window_actual, last_window_pred)

    st.markdown(' Latest prediction window')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Peak forecast', f'{peak_pred:,.0f} MW',
              help='Highest predicted load in the 24h window')
    c2.metric('Trough forecast', f'{trough_pred:,.0f} MW',
              help='Lowest predicted load in the 24h window')
    c3.metric('Average forecast', f'{avg_pred:,.0f} MW',
              help='Mean predicted load across the 24h window')
    c4.metric('Window MAE', f'{window_mae:,.0f} MW',
              help='Mean Absolute Error for this specific window')

    st.markdown('---')

    # Plot
    hours = list(range(1, MAX_PREDICTION_LENGTH + 1))
    fig = go.Figure()

    # Confidence band
    if last_q is not None and last_q.ndim == 2:
        q10 = last_q[:, 1]
        q90 = last_q[:, 5]
        fig.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=np.concatenate([q90, q10[::-1]]).tolist(),
            fill='toself', fillcolor='rgba(70,130,180,0.15)',
            line=dict(width=0), name='10th-90th percentile',
            hoverinfo='skip',
        ))

    fig.add_trace(go.Scatter(
        x=hours, y=last_window_actual,
        mode='lines+markers', name='Actual load',
        line=dict(color='steelblue', width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=last_window_pred,
        mode='lines+markers', name='TFT prediction',
        line=dict(color='tomato', width=2, dash='dot'),
        marker=dict(size=4),
    ))
    fig.update_layout(
        height=450, template='plotly_dark',
        xaxis_title=f'Hours ahead (1 = next hour, {MAX_PREDICTION_LENGTH} = {MAX_PREDICTION_LENGTH}h out)',
        yaxis_title='Load (MW)',
        hovermode='x unified',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    st.markdown(
        f'The blue line is the actual electricity demand. '
        f'The red dotted line is what the model predicted. '
        f'The shaded area shows the confidence interval '
        f'(the model is 80% confident the actual value falls within this range).'
    )

    st.markdown('---')

    # Recent load profile (last 7 days)
    st.markdown(' Recent load profile')
    n_recent = min(MAX_ENCODER_LENGTH, len(oos_actual))
    recent_actual = oos_actual[-n_recent:]
    recent_pred   = oos_pred[-n_recent:]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=recent_actual, name='Actual',
        line=dict(color='steelblue', width=1.5),
    ))
    fig2.add_trace(go.Scatter(
        y=recent_pred, name='TFT prediction',
        line=dict(color='tomato', width=1.5, dash='dot'),
    ))
    fig2.update_layout(
        height=350, template='plotly_dark',
        xaxis_title='Hour', yaxis_title='Load (MW)',
        hovermode='x unified',
    )
    st.plotly_chart(fig2, use_container_width=True)



elif page == 'Performance':
    st.title('Model Performance')
    st.markdown(
        'All metrics below are out-of-sample: the model never saw '
        'this data during training. This is the true measure of how well '
        'the model would perform in production.'
    )

    if not file_exists('oos_actual.npy'):
        st.warning('No results found. Run training first from the Pipeline page.')
        st.stop()

    oos_actual = np.load(os.path.join(SAVE_DIR, 'oos_actual.npy'))
    oos_pred   = np.load(os.path.join(SAVE_DIR, 'oos_predicted.npy'))

    mae_val   = mean_absolute_error(oos_actual, oos_pred)
    rmse_val  = np.sqrt(mean_squared_error(oos_actual, oos_pred))
    mape_val  = mape(oos_actual, oos_pred)
    r2_val    = r2_score(oos_actual, oos_pred)
    medae_val = np.median(np.abs(oos_actual - oos_pred))
    mean_load = oos_actual.mean()

    st.markdown(' Aggregate metrics (all walk-forward folds)')

    c1, c2, c3 = st.columns(3)
    c1.metric('MAE', f'{mae_val:,.0f} MW',
              help='Mean Absolute Error: average prediction error in MW')
    c2.metric('MAPE', f'{mape_val:.2f}%',
              help='Mean Absolute Percentage Error: error relative to actual load')
    c3.metric('R²', f'{r2_val:.4f}',
              help='R-squared: 1.0 = perfect, 0.0 = predicting the mean')

    c4, c5, c6 = st.columns(3)
    c4.metric('RMSE', f'{rmse_val:,.0f} MW',
              help='Root Mean Squared Error: penalizes large errors more')
    c5.metric('Median AE', f'{medae_val:,.0f} MW',
              help='Median Absolute Error: robust to outliers')
    c6.metric('Total OOS hours', f'{len(oos_actual):,}',
              help='Total number of hourly predictions across all folds')

    st.markdown(
        f'Average France load is {mean_load:,.0f} MW. '
        f'The model is off by {mae_val:,.0f} MW on average, '
        f'which is {mae_val / mean_load * 100:.2f}% of the mean load.'
    )

    # Confidence interval coverage
    if file_exists('oos_quantiles.npy'):
        oos_q = np.load(os.path.join(SAVE_DIR, 'oos_quantiles.npy'))
        q10_flat = oos_q[:, :, 1].flatten()[:len(oos_actual)]
        q90_flat = oos_q[:, :, 5].flatten()[:len(oos_actual)]
        coverage = np.mean((oos_actual >= q10_flat) & (oos_actual <= q90_flat)) * 100
        st.metric('Confidence interval coverage (10th-90th)', f'{coverage:.1f}%',
                  help='Target is 80%. If actual values fall within the predicted '
                       '10th-90th percentile range 80% of the time, the model is well calibrated.')

    st.markdown('---')

    # Actual vs predicted scatter
    st.markdown(' Actual vs predicted')
    st.markdown('Each dot is one hourly prediction. The red line is perfect accuracy.')

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=oos_actual, y=oos_pred, mode='markers',
        marker=dict(size=2, color='steelblue', opacity=0.15),
        name='Predictions',
    ))
    lims = [min(oos_actual.min(), oos_pred.min()),
            max(oos_actual.max(), oos_pred.max())]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode='lines', name='Perfect prediction',
        line=dict(color='red', dash='dash'),
    ))
    fig.update_layout(
        height=500, template='plotly_dark',
        xaxis_title='Actual load (MW)', yaxis_title='Predicted load (MW)',
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    # Error distribution
    st.markdown(' Error distribution')
    st.markdown(
        'Positive error = model predicted too high. '
        'Negative = predicted too low. '
        'A well-calibrated model centers around zero.'
    )

    errors = oos_pred - oos_actual
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=errors, nbinsx=100, marker_color='mediumseagreen', opacity=0.7,
    ))
    fig2.add_vline(x=0, line_dash='dash', line_color='red')
    fig2.update_layout(
        height=350, template='plotly_dark',
        xaxis_title='Error (MW)', yaxis_title='Count',
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f'Mean bias: {errors.mean():+,.0f} MW'
        f'({"overestimates" if errors.mean() > 0 else "underestimates"} on average). '
        f'Standard deviation: {errors.std():,.0f} MW.'
    )

    st.markdown('---')

    # Per-horizon breakdown
    st.markdown(' Accuracy by forecast horizon')
    st.markdown(
        f'Does the model get worse as it predicts further ahead? '
        f'Hour 1 is the next hour, hour {MAX_PREDICTION_LENGTH} is '
        f'{MAX_PREDICTION_LENGTH} hours out.'
    )

    n_windows = len(oos_actual) // MAX_PREDICTION_LENGTH
    if n_windows > 0:
        act_2d = oos_actual[:n_windows * MAX_PREDICTION_LENGTH].reshape(n_windows, MAX_PREDICTION_LENGTH)
        pred_2d = oos_pred[:n_windows * MAX_PREDICTION_LENGTH].reshape(n_windows, MAX_PREDICTION_LENGTH)

        horizon_mae = []
        horizon_mape = []
        for h in range(MAX_PREDICTION_LENGTH):
            h_mae = mean_absolute_error(act_2d[:, h], pred_2d[:, h])
            h_mape_val = mape(act_2d[:, h], pred_2d[:, h])
            horizon_mae.append(h_mae)
            horizon_mape.append(h_mape_val)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=list(range(1, MAX_PREDICTION_LENGTH + 1)),
            y=horizon_mae, marker_color='darkorange',
            name='MAE',
        ))
        fig3.update_layout(
            height=350, template='plotly_dark',
            xaxis_title='Hours ahead', yaxis_title='MAE (MW)',
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            f'Hour 1 MAE: {horizon_mae[0]:,.0f} MW ({horizon_mape[0]:.2f}%) — '
            f'Hour {MAX_PREDICTION_LENGTH} MAE: {horizon_mae[-1]:,.0f} MW '
            f'({horizon_mape[-1]:.2f}%)'
        )

    st.markdown('---')

    # Error by hour of day
    st.markdown(' Accuracy by time of day')
    st.markdown('Is the model better at predicting nighttime vs daytime demand?')

    n_hours = len(errors)
    hour_of_day = np.tile(np.arange(24), n_hours // 24 + 1)[:n_hours]
    err_df = pd.DataFrame({'hour': hour_of_day, 'abs_error': np.abs(errors)})
    hourly_mae = err_df.groupby('hour')['abs_error'].mean()

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=hourly_mae.index, y=hourly_mae.values,
        marker_color='orchid',
    ))
    fig4.update_layout(
        height=350, template='plotly_dark',
        xaxis_title='Hour of day (UTC)', yaxis_title='MAE (MW)',
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig4, use_container_width=True)

    best_hour  = hourly_mae.idxmin()
    worst_hour = hourly_mae.idxmax()
    st.markdown(
        f'Most accurate at {best_hour}:00 UTC ({hourly_mae[best_hour]:,.0f} MW MAE). '
        f'Least accurate at {worst_hour}:00 UTC ({hourly_mae[worst_hour]:,.0f} MW MAE).'
    )

    st.markdown('---')

    # Sample weeks
    st.markdown(' Sample prediction weeks')
    st.markdown('Three randomly selected weeks showing actual vs predicted load.')

    week_size = MAX_ENCODER_LENGTH
    n_weeks = min(3, len(oos_actual) // week_size)
    if n_weeks > 0:
        spacing = len(oos_actual) // (n_weeks + 1)
        for w in range(n_weeks):
            start = (w + 1) * spacing
            end = start + week_size
            if end > len(oos_actual):
                break
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(
                y=oos_actual[start:end], name='Actual',
                line=dict(color='steelblue', width=1.5),
            ))
            fig_w.add_trace(go.Scatter(
                y=oos_pred[start:end], name='TFT',
                line=dict(color='tomato', width=1.5, dash='dot'),
            ))
            local_mae = mean_absolute_error(oos_actual[start:end], oos_pred[start:end])
            local_mape_val = mape(oos_actual[start:end], oos_pred[start:end])
            fig_w.update_layout(
                height=280, template='plotly_dark',
                title=f'Week {w + 1} — MAE: {local_mae:,.0f} MW ({local_mape_val:.2f}%)',
                xaxis_title='Hour', yaxis_title='Load (MW)',
            )
            st.plotly_chart(fig_w, use_container_width=True)



elif page == 'Insights':
    st.title('What drives the predictions?')
    st.markdown(
        'Unlike black-box models, the Temporal Fusion Transformer reveals '
        'which features it pays attention to and which past hours '
        'influence its predictions the most. This builds trust: if the model '
        'says temperature and yesterday\'s load matter most, that matches '
        'how grid operators actually think.'
    )

    if not file_exists('tft_interpretation.pkl'):
        st.warning(
            'No interpretability data found. Run training from the Pipeline page. '
            'The interpretation is extracted automatically at the end of training.'
        )
        st.stop()

    import joblib
    interp = joblib.load(os.path.join(SAVE_DIR, 'tft_interpretation.pkl'))

    # Variable importance
    for key, title, description in [
        ('encoder_variables', 'Encoder variable importance',
         'These are the input features from the past 7 days. '
         'Higher importance means the feature has more influence on the prediction.'),
        ('decoder_variables', 'Decoder variable importance',
         'These are features known in the future (calendar, weather forecasts). '
         'They tell the model about conditions during the predicted 24h window.'),
        ('static_variables', 'Static variable importance',
         'These are constant features across the time series (e.g. country identifier).'),
    ]:
        if key not in interp:
            continue

        data = interp[key]
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            continue

        st.markdown(f' {title}')
        st.markdown(description)

        # Try to get variable names from the model
        n_vars = len(data)
        labels = [f'Variable {i}' for i in range(n_vars)]

        # Sort by importance
        sorted_idx = np.argsort(data)[::-1]
        sorted_data = data[sorted_idx]
        sorted_labels = [labels[i] for i in sorted_idx]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_data[:15],
            y=sorted_labels[:15],
            orientation='h',
            marker_color='steelblue',
        ))
        fig.update_layout(
            height=max(300, min(15, n_vars) * 30),
            template='plotly_dark',
            xaxis_title='Importance',
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Attention weights
    if 'attention' in interp:
        st.markdown(' Attention pattern')
        st.markdown(
            f'This shows which of the past {MAX_ENCODER_LENGTH} hours '
            f'the model focuses on when making predictions. '
            f'Spikes indicate time steps the model considers most informative. '
            f'You should see peaks at -24h (same hour yesterday) and '
            f'-168h (same hour last week) if the model learned the right patterns.'
        )

        attn = interp['attention']
        if isinstance(attn, np.ndarray):
            # Average down to 1D if needed
            while attn.ndim > 1:
                attn = attn.mean(axis=0)

            hours_back = list(range(-len(attn), 0))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours_back, y=attn,
                mode='lines', line=dict(color='darkorange', width=2),
            ))

            # Mark key lags
            for lag, label in [(-24, '24h ago'), (-168, '7 days ago')]:
                if abs(lag) <= len(attn):
                    idx = len(attn) + lag
                    if 0 <= idx < len(attn):
                        fig.add_vline(
                            x=lag, line_dash='dot', line_color='gray',
                            annotation_text=label, annotation_position='top',
                        )

            fig.update_layout(
                height=350, template='plotly_dark',
                xaxis_title='Hours before prediction',
                yaxis_title='Attention weight',
            )
            st.plotly_chart(fig, use_container_width=True)

            peak_hour = hours_back[np.argmax(attn)]
            st.markdown(f'The model pays the most attention to {abs(peak_hour)} hours ago.')

    st.markdown('---')
    st.markdown(
        'Why this matters for energy companies: grid operators need to trust '
        'AI predictions before acting on them. Showing that the model focuses on '
        'yesterday\'s load at the same hour and on temperature aligns with '
        'decades of operational experience. An uninterpretable model that '
        'achieves the same accuracy but can\'t explain itself would not be '
        'deployed in production.'
    )


elif page == 'Pipeline':
    st.title('Data & Training Pipeline')
    st.markdown(
        'Run each step in order. The pipeline fetches real electricity demand '
        'from ENTSO-E and weather from Open-Meteo, trains a Temporal Fusion '
        'Transformer with walk-forward cross validation, and generates predictions.'
    )

    api_key_set = bool(os.environ.get('ENTSOE_API_KEY', ''))

    if not api_key_set:
        st.error(
            'ENTSOE_API_KEY environment variable is not set. '
            'Set it before launching: `export ENTSOE_API_KEY="your-key"` '
            'then restart Streamlit.'
        )

    # Step 1: Data
    st.markdown('---')
    st.markdown(' Step 1: Fetch data')
    st.markdown(
        f'Downloads hourly electricity load for {COUNTRY_CODE} from ENTSO-E '
        f'(from {START_DATE} to today) and hourly weather for Paris from Open-Meteo. '
        f'Merges them into a single dataset.'
    )

    data_ready = file_exists('france_load_weather.parquet')
    if data_ready:
        df_info = pd.read_parquet(os.path.join(SAVE_DIR, 'france_load_weather.parquet'))
        st.success(
            f'Dataset exists: {len(df_info):,} hours '
            f'({df_info.index[0].strftime("%Y-%m-%d")} to '
            f'{df_info.index[-1].strftime("%Y-%m-%d")})'
        )

    if st.button('Run data.py', type='primary', disabled=not api_key_set):
        with st.spinner('Fetching data from ENTSO-E and Open-Meteo (may take a few minutes)...'):
            output = run_script('data.py')
        st.code(output, language='text')
        if 'done!' in output:
            st.success('Data fetched and saved.')
        else:
            st.error('Something went wrong. Check the output above.')

    # Step 2: Train
    st.markdown('---')
    st.markdown(' Step 2: Train model')
    st.markdown(
        f'Trains a Temporal Fusion Transformer using walk-forward cross validation. '
        f'Each fold trains on {WF_TRAIN_SIZE // 24} days, validates on {WF_VAL_SIZE // 24} days, '
        f'and tests on {WF_TEST_SIZE // 24} days. The window advances by '
        f'{WF_STEP_SIZE // 24} days per fold. '
        f'Also runs XGBoost as a baseline comparison.'
    )
    st.markdown(
        f'Training runs for up to {EPOCHS} epochs per fold with '
        f'early stopping (patience={PATIENCE}). This may take '
        f'10-30 minutes depending on your GPU.'
    )

    model_ready = file_exists('tft_model.pth')
    if model_ready:
        st.success('Trained model found (tft_model.pth)')

    if st.button('Run train.py', type='primary', disabled=not data_ready):
        with st.spinner('Training TFT with walk-forward CV (this takes a while)...'):
            output = run_script('train.py')
        st.code(output, language='text')
        if 'done!' in output:
            st.success('Training complete. Check Performance and Insights pages.')
        else:
            st.error('Training may have failed. Check the output above.')

    # Step 3: Predict
    st.markdown('---')
    st.markdown(' Step 3: Run live prediction')
    st.markdown(
        f'Fetches the most recent {MAX_ENCODER_LENGTH // 24} days of data, '
        f'runs the trained model, and prints a {MAX_PREDICTION_LENGTH}-hour '
        f'forecast with confidence intervals.'
    )

    if st.button('Run predict.py', type='primary', disabled=not model_ready):
        with st.spinner('Fetching latest data and running inference...'):
            output = run_script('predict.py')
        st.code(output, language='text')
        if 'done!' in output:
            st.success('Prediction complete.')
        else:
            st.error('Prediction may have failed. Check the output above.')

    # Training results image
    st.markdown('---')
    st.markdown(' Training results')
    results_img = os.path.join(SAVE_DIR, 'training_results.png')
    if os.path.exists(results_img):
        st.image(results_img, caption='Walk-forward OOS results')
    else:
        st.info('No training results plot yet. Run training first.')