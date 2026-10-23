import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

if __package__ in {None, ''}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import forcing_names, set_config_path, validate_dap_weights

VARIABLES = (
    ('thetao', 'degC', True),
    ('so', 'psu', True),
    ('uo', 'm s-1', True),
    ('vo', 'm s-1', True),
    ('zos', 'm', False),
)
METHODS = ('persistence', 'monthly_climatology', 'anomaly_persistence', 'linear_tendency')
ALLOWED_UNITS = {'degC', 'psu', 'm s-1', 'm'}


def append_metrics(rows, method, init_time, lead_day, name, unit, prediction, truth, initial, depth_values):
    valid = np.isfinite(initial) & np.isfinite(truth) & np.isfinite(prediction)
    error2 = np.where(valid, (prediction - truth) ** 2, 0.0)
    if depth_values is None:
        count = valid.sum()
        if count == 0:
            return
        rows.append({
            'method': method,
            'init_time': init_time.date().isoformat(),
            'lead_day': lead_day,
            'variable': name,
            'depth_m': np.nan,
            'unit': unit,
            'rmse': float(np.sqrt(error2.sum() / count)),
        })
        return
    counts = valid.sum(axis=(-2, -1))
    sums = error2.sum(axis=(-2, -1))
    for index, depth_m in enumerate(depth_values):
        if counts[index] == 0:
            continue
        rows.append({
            'method': method,
            'init_time': init_time.date().isoformat(),
            'lead_day': lead_day,
            'variable': name,
            'depth_m': float(depth_m),
            'unit': unit,
            'rmse': float(np.sqrt(sums[index] / counts[index])),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--dap-weights')
    parser.add_argument('--include-dap', action='store_true')
    args = parser.parse_args()
    if args.include_dap and not args.dap_weights:
        raise ValueError('--include-dap requires --dap-weights. DAP must be calibrated on validation before evaluation.')

    config_path = Path(args.config)
    config = set_config_path(yaml.safe_load(config_path.read_text(encoding='utf-8-sig')), config_path)
    output_dir = Path(config['output_dir'])
    run_dir = Path(config['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    ocean = xr.open_dataset(output_dir / 'ocean_state_daily.nc')
    samples = pd.read_csv(output_dir / 'samples.csv', parse_dates=['init_time'])
    init_times = samples.query('split == @args.split').init_time.tolist()
    if not init_times:
        raise ValueError(f'No {args.split} samples are available.')

    train_end = pd.Timestamp(config['splits']['train_end'])
    train_ocean = ocean.sel(time=slice(None, train_end))
    climatology = train_ocean.groupby('time.month').mean('time', skipna=True)
    dap_gamma = None
    if args.dap_weights:
        payload = json.loads(Path(args.dap_weights).read_text(encoding='utf-8-sig'))
        dap_gamma = validate_dap_weights(payload, config, forcing_names(config))
    rows = []

    for init_time in init_times:
        init_time = pd.Timestamp(init_time)
        previous_time = init_time - pd.Timedelta(days=1)
        initial_fields = {name: ocean[name].sel(time=init_time).values for name, _, _ in VARIABLES}
        previous_fields = {name: ocean[name].sel(time=previous_time).values for name, _, _ in VARIABLES}
        climatology_initial = {name: climatology[name].sel(month=init_time.month).values for name, _, _ in VARIABLES}
        for lead_day in range(1, 11):
            target_time = init_time + pd.Timedelta(days=lead_day)
            for name, unit, has_depth in VARIABLES:
                initial = initial_fields[name]
                previous = previous_fields[name]
                truth = ocean[name].sel(time=target_time).values
                clim_initial = climatology_initial[name]
                clim_target = climatology[name].sel(month=target_time.month).values
                predictions = {
                    'persistence': initial,
                    'monthly_climatology': clim_target,
                    'anomaly_persistence': clim_target + (initial - clim_initial),
                    'linear_tendency': initial + lead_day * (initial - previous),
                }
                if dap_gamma is not None:
                    predictions['damped_anomaly_persistence'] = clim_target + float(dap_gamma[name][lead_day - 1]) * (initial - clim_initial)
                depth_values = ocean.depth.values if has_depth else None
                for method, prediction in predictions.items():
                    append_metrics(rows, method, init_time, lead_day, name, unit, prediction, truth, initial, depth_values)

    metrics = pd.DataFrame(rows)
    bad_units = sorted(set(metrics['unit'].dropna().astype(str)) - ALLOWED_UNITS)
    if bad_units:
        raise ValueError(f'Unexpected unit values in statistical baseline output: {bad_units}')
    suffix = '' if args.split == 'test' else f'_{args.split}'
    depth_path = run_dir / f'statistical_baselines_by_variable_depth{suffix}.csv'
    summary_path = run_dir / f'statistical_baselines_by_variable{suffix}.csv'
    metrics.to_csv(depth_path, index=False)
    summary = metrics.groupby(['method', 'lead_day', 'variable', 'unit'], as_index=False)['rmse'].mean()
    summary.to_csv(summary_path, index=False)
    print({'summary': str(summary_path), 'depth_metrics': str(depth_path), 'climatology_train_end': train_end.date().isoformat()})


if __name__ == '__main__':
    main()
