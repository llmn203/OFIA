import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

if __package__ in {None, ''}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import architecture_name, build_model, denorm, features_from_history, forcing_array, forcing_history_days, forcing_history_mode, forcing_names, history_days, input_layout_metadata, load_stats, model_channels_in, set_config_path, state_array, static_array, update_rolling_histories, validate_blend_weights, validate_checkpoint_interface

VARIABLES = (
    ('thetao', slice(0, 33), 'degC'),
    ('so', slice(33, 66), 'psu'),
    ('uo', slice(66, 99), 'm s-1'),
    ('vo', slice(99, 132), 'm s-1'),
    ('zos', slice(132, 133), 'm'),
)


def normalized_monthly_climatology(ocean, stats, train_end):
    train_ocean = ocean.sel(time=slice(None, train_end))
    if train_ocean.sizes.get('time', 0) == 0:
        raise ValueError(f'No ocean samples are available for climatology before train_end={train_end}.')
    climatology = train_ocean.groupby('time.month').mean('time', skipna=True)
    template = np.asarray(ocean['thetao'].isel(time=0).values)
    monthly = np.full((13, 133, template.shape[-2], template.shape[-1]), np.nan, dtype=np.float32)
    for month in climatology.month.values:
        month_index = int(month)
        for name, channels, _ in VARIABLES:
            values = np.asarray(climatology[name].sel(month=month).values, dtype=np.float32)
            normalized = (values - stats['ocean'][name]['mean']) / max(stats['ocean'][name]['std'], 1e-6)
            if name == 'zos':
                monthly[month_index, channels.start] = normalized
            else:
                monthly[month_index, channels] = normalized
    return monthly, pd.Timestamp(train_ocean.time.min().values), pd.Timestamp(train_ocean.time.max().values)


def climatological_acc(prediction, truth, climatology, valid, eps=1e-12):
    valid = valid & np.isfinite(prediction) & np.isfinite(truth) & np.isfinite(climatology)
    if not np.any(valid):
        return np.nan
    # ACC is the spatial Pearson correlation between forecast anomalies and
    # truth anomalies relative to the training-period monthly climatology.
    # The climatology is computed exclusively from data up to train_end to
    # prevent validation/test leakage.
    forecast_anomaly = prediction[valid] - climatology[valid]
    truth_anomaly = truth[valid] - climatology[valid]
    forecast_centered = forecast_anomaly - np.mean(forecast_anomaly)
    truth_centered = truth_anomaly - np.mean(truth_anomaly)
    denominator = float(np.sqrt(np.sum(forecast_centered ** 2) * np.sum(truth_centered ** 2)))
    if denominator <= eps or not np.isfinite(denominator):
        return np.nan
    return float(np.sum(forecast_centered * truth_centered) / denominator)


def load_blend_weights(path, config, checkpoint_path, checkpoint, force_vars, channels_in):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return validate_blend_weights(payload, config, checkpoint_path, checkpoint, force_vars, channels_in)


def physical_prediction_dataset(predictions, ocean, static, stats):
    data = {}
    wet = static.wet_mask_3d.values.astype(bool)
    for name, channels, _ in VARIABLES:
        values = denorm(predictions[:, channels], stats['ocean'][name])
        if name == 'zos':
            data[name] = (('lead_day', 'latitude', 'longitude'), np.where(wet[0][None], values[:, 0], np.nan))
        else:
            data[name] = (('lead_day', 'depth', 'latitude', 'longitude'), np.where(wet[None], values, np.nan))
    return xr.Dataset(
        data,
        coords={
            'lead_day': np.arange(1, predictions.shape[0] + 1),
            'depth': ocean.depth.values,
            'latitude': ocean.latitude.values,
            'longitude': ocean.longitude.values,
        },
        attrs={'description': 'Autoregressive hindcast prediction in physical units.'},
    )


def save_surface_error_map(prediction, truth, ocean, stats, surface_mask, output_path):
    pred_c = np.where(surface_mask, denorm(prediction[0], stats['ocean']['thetao']), np.nan)
    truth_c = np.where(surface_mask, denorm(truth[0], stats['ocean']['thetao']), np.nan)
    error = pred_c - truth_c
    vmax = float(np.nanpercentile(np.abs(error), 98))
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    extent = [float(ocean.longitude.min()), float(ocean.longitude.max()), float(ocean.latitude.min()), float(ocean.latitude.max())]
    panels = ((pred_c, 'Predicted surface thetao'), (truth_c, 'Observed surface thetao'), (error, 'Prediction minus observation'))
    for ax, (field, title) in zip(axes, panels):
        if title == 'Prediction minus observation':
            image = ax.imshow(field, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            label = 'degC'
        else:
            image = ax.imshow(field, origin='lower', extent=extent, cmap='turbo', aspect='auto')
            label = 'degC'
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.colorbar(image, ax=ax, label=label)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--blend-weights')
    args = parser.parse_args()

    config_path = Path(args.config)
    config = set_config_path(yaml.safe_load(config_path.read_text(encoding='utf-8-sig')), config_path)
    force_vars = forcing_names(config)
    history_days_value = history_days(config)
    forcing_history_days_value = forcing_history_days(config)
    forcing_history_mode_value = forcing_history_mode(config)
    layout = input_layout_metadata(config, force_vars)
    architecture = architecture_name(config)
    output_dir = Path(config['output_dir'])
    run_dir = Path(config['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    ocean = xr.open_dataset(output_dir / 'ocean_state_daily.nc')
    forcing = xr.open_dataset(output_dir / 'forcing_daily.nc')
    static = xr.open_dataset(output_dir / 'static.nc')
    normalization_path = output_dir / 'normalization.json'
    stats = load_stats(normalization_path)
    train_end = pd.Timestamp(config['splits']['train_end'])
    monthly_climatology, climatology_start, climatology_end = normalized_monthly_climatology(ocean, stats, train_end)
    print({
        'stage': 'monthly_climatology',
        'climatology_start': climatology_start.date().isoformat(),
        'climatology_end': climatology_end.date().isoformat(),
        'train_end': train_end.date().isoformat(),
        'space': 'normalized',
    }, flush=True)
    samples = pd.read_csv(output_dir / 'samples.csv', parse_dates=['init_time'])
    test_times = samples.query("split == 'test'").init_time.tolist()
    if not test_times:
        raise ValueError('No test samples are available for evaluation.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    channels_in = model_channels_in(history_days_value, force_vars, forcing_history_days_value)
    validate_checkpoint_interface(checkpoint, config, force_vars, channels_in, normalization_path)
    net = build_model(architecture, channels_in, 133, config['model']['width'], config.get('model', {})).to(device)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    blend_weights = load_blend_weights(args.blend_weights, config, args.checkpoint, checkpoint, force_vars, channels_in)
    suffix = '_blended' if blend_weights is not None else ''
    print({'stage': 'evaluation_input_layout', **layout}, flush=True)

    rows = []
    saved = False
    with torch.no_grad():
        for init_time in test_times:
            init_time = pd.Timestamp(init_time)
            history = []
            for offset in range(history_days_value - 1, -1, -1):
                state, _ = state_array(ocean, init_time - pd.Timedelta(days=offset), stats)
                history.append(state)
            forcing_history = None
            if forcing_history_days_value:
                forcing_history = [
                    forcing_array(forcing, init_time - pd.Timedelta(days=offset), stats, force_vars)
                    for offset in range(forcing_history_days_value - 1, -1, -1)
                ]
            current = history[-1]
            _, initial_mask = state_array(ocean, init_time, stats)
            static_features = static_array(static, stats)
            initial_state = current.copy()
            predictions = []
            truths = []
            for lead_day in range(1, 11):
                target_time = init_time + pd.Timedelta(days=lead_day)
                lead_force = forcing_array(forcing, target_time, stats, force_vars)
                input_features = features_from_history(
                    history,
                    lead_force,
                    static_features,
                    forcing_history,
                )[None]
                raw_prediction = (current + net(torch.as_tensor(input_features, dtype=torch.float32, device=device)).cpu().numpy()[0]) * initial_mask
                alpha = blend_weights[lead_day - 1, :, None, None] if blend_weights is not None else 1.0
                prediction = initial_state + alpha * (raw_prediction - initial_state)
                truth, target_mask = state_array(ocean, target_time, stats)
                target_climatology = monthly_climatology[int(target_time.month)]
                for name, channels, unit in VARIABLES:
                    standard_deviation = stats['ocean'][name]['std']
                    for channel_offset, channel_index in enumerate(range(channels.start, channels.stop)):
                        valid = initial_mask[channel_index].astype(bool) & target_mask[channel_index].astype(bool)
                        if not np.any(valid):
                            continue
                        predicted_values = prediction[channel_index][valid]
                        truth_values = truth[channel_index][valid]
                        initial_values = initial_state[channel_index][valid]
                        error = predicted_values - truth_values
                        persistence_error = initial_values - truth_values
                        normalized_rmse = float(np.sqrt(np.mean(error ** 2)))
                        normalized_persistence = float(np.sqrt(np.mean(persistence_error ** 2)))
                        mae = float(np.mean(np.abs(error)) * standard_deviation)
                        bias = float(np.mean(error) * standard_deviation)
                        ss_res = float(np.sum(error ** 2))
                        truth_centered = truth_values - np.mean(truth_values)
                        ss_tot = float(np.sum(truth_centered ** 2))
                        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan
                        acc = climatological_acc(prediction[channel_index], truth[channel_index], target_climatology[channel_index], valid)
                        depth_m = float(ocean.depth.values[channel_offset]) if name != 'zos' else np.nan
                        rows.append({
                            'init_time': init_time.date().isoformat(),
                            'lead_day': lead_day,
                            'variable': name,
                            'depth_m': depth_m,
                            'unit': unit,
                            'rmse': normalized_rmse * standard_deviation,
                            'persistence_rmse': normalized_persistence * standard_deviation,
                            'normalized_rmse': normalized_rmse,
                            'normalized_persistence_rmse': normalized_persistence,
                            'mae': mae,
                            'bias': bias,
                            'r2': r2,
                            'acc': acc,
                        })
                predictions.append(prediction)
                truths.append(truth)
                history, forcing_history = update_rolling_histories(history, raw_prediction, forcing_history, lead_force)
                current = raw_prediction

            if not saved:
                prediction_stack = np.stack(predictions)
                physical_prediction_dataset(prediction_stack, ocean, static, stats).to_netcdf(run_dir / ('first_test_prediction' + suffix + '.nc'))
                save_surface_error_map(prediction_stack[-1], np.stack(truths)[-1], ocean, stats, static.wet_mask_3d.isel(depth=0).values.astype(bool), run_dir / ('first_test_lead10_surface_thetao' + suffix + '.png'))
                saved = True

    metrics = pd.DataFrame(rows)
    metrics['blend_applied'] = blend_weights is not None
    metrics.to_csv(run_dir / ('metrics_by_variable_depth' + suffix + '.csv'), index=False)
    summary = metrics.groupby(['lead_day', 'variable', 'unit', 'blend_applied'], as_index=False)[['rmse', 'persistence_rmse', 'mae', 'bias', 'r2', 'acc']].mean()
    summary.to_csv(run_dir / ('metrics_by_variable' + suffix + '.csv'), index=False)
    print(run_dir / ('metrics_by_variable' + suffix + '.csv'))


if __name__ == '__main__':
    main()
