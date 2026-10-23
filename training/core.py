import hashlib
import json
from pathlib import Path

import numpy as np

from models import ConvGRUUNetV1, ConvLSTMUNetV1, LSTMBottleneckV1, MultiScaleResUNetV2, ResidualUNet, SimVPV1, TemporalTransformerV1

STATE_3D=('thetao','so','uo','vo'); STATE_2D=('zos',); FORCE=('tau_x','tau_y','qnet','freshwater','t2m','msl','sin_doy','cos_doy')
STATE_CHANNELS=133; STATIC_CHANNELS=4; DEFAULT_ARCHITECTURE='residual_unet_v1'; DEFAULT_HISTORY_DAYS=2

def architecture_name(config):
 return str(config.get('architecture',config.get('model',{}).get('architecture',DEFAULT_ARCHITECTURE)))
def history_days(config):
 value=int(config.get('history_days',config.get('model',{}).get('history_days',DEFAULT_HISTORY_DAYS)))
 if value<2:
  raise ValueError('history_days must be at least 2.')
 return value
def forcing_history_days(config):
 value=int(config.get('forcing_history_days',config.get('model',{}).get('forcing_history_days',0)))
 if value<0:
  raise ValueError('forcing_history_days must be non-negative.')
 if value and value!=history_days(config):
  raise ValueError('forcing_history_days must equal history_days so ocean and forcing histories stay time-aligned.')
 return value
def forcing_history_mode(config):
 days=forcing_history_days(config)
 mode=str(config.get('forcing_history_mode','rolling' if days else 'none')).strip().lower()
 if not days:
  if mode not in {'none','disabled'}:
   raise ValueError('forcing_history_mode must be none when forcing_history_days=0.')
  return 'none'
 if mode!='rolling':
  raise ValueError('Only forcing_history_mode=rolling is supported when forcing_history_days>0.')
 return mode
def forcing_names(config):
 """Return the configured forcing interface, preserving the legacy MVP default."""
 names=tuple(config.get('forcing_variables',FORCE))
 if not names or len(set(names))!=len(names):
  raise ValueError('forcing_variables must be a non-empty list of unique names.')
 return names
def model_channels_in(history_days_value,force_vars,forcing_history_days_value=0):
 force_channels=len(tuple(force_vars))
 return int(history_days_value)*STATE_CHANNELS+int(forcing_history_days_value)*force_channels+force_channels+STATIC_CHANNELS
def input_strategy(config):
 hdays=history_days(config); fdays=forcing_history_days(config)
 if not fdays:
  return f'ocean_hist_l{hdays}_forcing_lead_static'
 return f'ocean_hist_l{hdays}_forcing_hist_l{fdays}_rolling_forcing_lead_static'
def input_layout_metadata(config,force_vars):
 hdays=history_days(config); fdays=forcing_history_days(config); force_channels=len(tuple(force_vars))
 strategy=input_strategy(config)
 declared=config.get('input_strategy')
 if declared is not None and str(declared)!=strategy:
  raise ValueError(f'Configured input_strategy={declared} does not match derived strategy={strategy}.')
 return {
  'input_strategy':strategy,
  'history_days':hdays,
  'forcing_history_days':fdays,
  'forcing_history_mode':forcing_history_mode(config),
  'state_channels':STATE_CHANNELS,
  'forcing_channels':force_channels,
  'static_channels':STATIC_CHANNELS,
  'ocean_history_channels':hdays*STATE_CHANNELS,
  'forcing_history_channels':fdays*force_channels,
  'lead_forcing_channels':force_channels,
  'input_channels':model_channels_in(hdays,force_vars,fdays),
 }
def build_model(architecture,cin,cout=STATE_CHANNELS,width=24,model_config=None):
 model_config=model_config or {}
 if architecture=='residual_unet_v1':
  return ResidualUNet(cin,cout,width)
 if architecture=='multiscale_resunet_v2':
  blocks_per_stage=int(model_config.get('blocks_per_stage',1))
  mid_blocks=int(model_config.get('mid_blocks',1))
  return MultiScaleResUNetV2(cin,cout,width,blocks_per_stage=blocks_per_stage,mid_blocks=mid_blocks)
 if architecture=='lstm_bottleneck_v1':
  return LSTMBottleneckV1(cin,cout,width)
 if architecture=='convlstm_unet_v1':
  return ConvLSTMUNetV1(cin,cout,width)
 if architecture=='convgru_unet_v1':
  return ConvGRUUNetV1(cin,cout,width)
 if architecture=='simvp_v1':
  return SimVPV1(cin,cout,width)
 if architecture=='temporal_transformer_v1':
  return TemporalTransformerV1(cin,cout,width)
 raise ValueError(f'Unsupported architecture: {architecture}')
def load_stats(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for chunk in iter(lambda: handle.read(1024*1024),b''):
   h.update(chunk)
 return h.hexdigest()
def config_sha256(path): return sha256_file(path)
def set_config_path(config,path):
 config['_config_path']=str(Path(path))
 return config
def current_config_sha256(config):
 path=config.get('_config_path')
 if not path:
  raise ValueError('Config path is not attached; call set_config_path(config, path) before strict metadata validation.')
 return config_sha256(path)
def canonical_json_sha256(payload):
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),default=str).encode('utf-8')).hexdigest()
def training_signature(config,normalization_sha256):
 model=config.get('model',{})
 allowed_model_changes={'epochs','batch_size','val_batch_size','grad_accum','progress_interval','val_progress_interval','max_train_samples','max_val_samples'}
 signed_model={key:value for key,value in model.items() if key not in allowed_model_changes}
 payload={
  'architecture':architecture_name(config),
  'history_days':history_days(config),
  'forcing_variables':list(forcing_names(config)),
  'model':signed_model,
  'loss':config.get('loss',{}),
  'output_dir':config.get('output_dir'),
  'splits':config.get('splits',{}),
 'normalization_sha256':normalization_sha256,
 }
 if forcing_history_days(config):
  payload['forcing_history_days']=forcing_history_days(config)
  payload['forcing_history_mode']=forcing_history_mode(config)
  payload['input_strategy']=input_strategy(config)
 return canonical_json_sha256(payload)
def norm(x,stat): return (x-stat['mean'])/max(stat['std'],1e-6)
def denorm(x,stat): return x*max(stat['std'],1e-6)+stat['mean']
def transform_forcing(x,stat):
 transform=stat.get('transform','identity')
 if transform=='identity': return x
 if transform=='log1p_nonnegative': return np.log1p(np.maximum(x,0))
 raise ValueError(f'Unsupported forcing transform: {transform}')
def state_array(ds,t,stats):
 a=[]; m=[]
 for v in STATE_3D:
  x=ds[v].sel(time=t).values; valid=np.isfinite(x)
  values=norm(np.where(valid,x,0),stats['ocean'][v]); a.append(np.where(valid,values,0).astype('float32')); m.append(valid.astype('float32'))
 x=ds.zos.sel(time=t).values; valid=np.isfinite(x)
 values=norm(np.where(valid,x,0),stats['ocean']['zos']); a.append(np.where(valid,values,0).astype('float32')[None]); m.append(valid.astype('float32')[None])
 return np.concatenate(a),np.concatenate(m)
def forcing_array(ds,t,stats,names=FORCE):
 reference=ds[names[0]].sel(time=t).values
 out=[]
 for v in names:
  if v not in ds or v not in stats['forcing']:
   raise KeyError(f'Missing forcing variable: {v}')
  x=ds[v].sel(time=t).values
  if np.ndim(x)==0:
   x=np.full_like(reference,float(x),dtype=np.float32)
  out.append(norm(transform_forcing(np.nan_to_num(x),stats['forcing'][v]),stats['forcing'][v]))
 return np.stack(out)
def static_array(ds,stats):
 wet=ds.wet_mask_3d.isel(depth=0).values.astype('float32'); bathy=norm(np.nan_to_num(ds.bathymetry.values),stats['static']['bathymetry']); lon=np.broadcast_to(ds.lon_sin.values,(ds.latitude.size,ds.longitude.size)); lat=np.broadcast_to(ds.lat_sin.values[:,None],(ds.latitude.size,ds.longitude.size)); return np.stack([wet,bathy,lon,lat])
def features(prev,cur,forcing,static): return np.concatenate([prev,cur,forcing,static])
def input_feature_parts(history,forcing,static,forcing_history=None):
 parts=list(history)
 if forcing_history is not None:
  if len(forcing_history)!=len(history):
   raise ValueError(f'Forcing history length {len(forcing_history)} must equal ocean history length {len(history)}.')
  parts.extend(forcing_history)
 parts.extend([forcing,static])
 return parts
def features_from_history(history,forcing,static,forcing_history=None):
 return np.concatenate(input_feature_parts(history,forcing,static,forcing_history))
def update_rolling_histories(history,prediction,forcing_history=None,forcing_lead=None):
 next_history=history[1:]+[prediction]
 if forcing_history is None:
  return next_history,None
 if forcing_lead is None:
  raise ValueError('forcing_lead is required when rolling forcing history is enabled.')
 if len(forcing_history)!=len(history):
  raise ValueError('Ocean and forcing history windows must have the same length before rolling update.')
 return next_history,forcing_history[1:]+[forcing_lead]
def checkpoint_architecture(checkpoint): return str(checkpoint.get('architecture',DEFAULT_ARCHITECTURE))
def checkpoint_history_days(checkpoint): return int(checkpoint.get('history_days',DEFAULT_HISTORY_DAYS))
def checkpoint_forcing_history_days(checkpoint): return int(checkpoint.get('forcing_history_days',0))
def validate_checkpoint_interface(checkpoint,config,force_vars,cin,normalization_path=None):
 expected_arch=architecture_name(config); actual_arch=checkpoint_architecture(checkpoint)
 if actual_arch!=expected_arch:
  raise ValueError(f'Checkpoint architecture={actual_arch} does not match config architecture={expected_arch}.')
 expected_history=history_days(config); actual_history=checkpoint_history_days(checkpoint)
 if actual_history!=expected_history:
  raise ValueError(f'Checkpoint history_days={actual_history} does not match config history_days={expected_history}.')
 expected_forcing_history=forcing_history_days(config); actual_forcing_history=checkpoint_forcing_history_days(checkpoint)
 if actual_forcing_history!=expected_forcing_history:
  raise ValueError(f'Checkpoint forcing_history_days={actual_forcing_history} does not match config forcing_history_days={expected_forcing_history}.')
 if expected_forcing_history:
  expected_mode=forcing_history_mode(config); actual_mode=str(checkpoint.get('forcing_history_mode','none'))
  if actual_mode!=expected_mode:
   raise ValueError(f'Checkpoint forcing_history_mode={actual_mode} does not match config forcing_history_mode={expected_mode}.')
 checkpoint_force_vars=tuple(checkpoint.get('forcing_variables',force_vars))
 if checkpoint_force_vars!=tuple(force_vars):
  raise ValueError(f'Checkpoint forcing interface {checkpoint_force_vars} does not match config {tuple(force_vars)}.')
 if int(checkpoint['cin'])!=int(cin):
  raise ValueError(f'Checkpoint cin={checkpoint["cin"]} does not match current cin={cin}.')
 if normalization_path is not None and 'normalization_sha256' in checkpoint:
  current=sha256_file(normalization_path)
  if checkpoint['normalization_sha256']!=current:
   raise ValueError(f'Checkpoint normalization_sha256 does not match current normalization.json: {checkpoint["normalization_sha256"]} != {current}.')
def validate_training_signature(checkpoint,config,normalization_path):
 expected=training_signature(config,sha256_file(normalization_path))
 actual=checkpoint.get('training_signature')
 if actual is None:
  print({'warning':'legacy_checkpoint_missing_training_signature','resume_check':'interface_only'},flush=True)
  return
 if actual!=expected:
  raise ValueError(f'Checkpoint training_signature does not match current config/data signature: {actual} != {expected}.')
def validate_blend_weights(payload,config,checkpoint_path,checkpoint,expected_force_vars,expected_cin):
 if payload.get('calibration_type')!='persistence_blend':
  raise ValueError('Blend weights calibration_type must be persistence_blend.')
 if payload.get('fit_split')!='val':
  raise ValueError('Blend weights fit_split must be val.')
 expected_arch=architecture_name(config)
 if payload.get('architecture')!=expected_arch:
  raise ValueError(f'Blend architecture={payload.get("architecture")} does not match config architecture={expected_arch}.')
 expected_history=history_days(config)
 if int(payload.get('history_days',-1))!=expected_history:
  raise ValueError(f'Blend history_days={payload.get("history_days")} does not match config history_days={expected_history}.')
 if tuple(payload.get('forcing_variables',()))!=tuple(expected_force_vars):
  raise ValueError(f'Blend forcing_variables={payload.get("forcing_variables")} does not match config {list(expected_force_vars)}.')
 if int(payload.get('cin',-1))!=int(expected_cin):
  raise ValueError(f'Blend cin={payload.get("cin")} does not match current cin={expected_cin}.')
 expected_config_hash=current_config_sha256(config)
 if payload.get('config_sha256')!=expected_config_hash:
  raise ValueError(f'Blend config_sha256 does not match current config: {payload.get("config_sha256")} != {expected_config_hash}.')
 expected_checkpoint_hash=sha256_file(checkpoint_path)
 if payload.get('checkpoint_sha256')!=expected_checkpoint_hash:
  raise ValueError(f'Blend checkpoint_sha256 does not match current checkpoint file: {payload.get("checkpoint_sha256")} != {expected_checkpoint_hash}.')
 validate_checkpoint_interface(checkpoint,config,expected_force_vars,expected_cin)
 alphas=np.asarray(payload.get('alphas'),dtype=np.float32)
 if alphas.shape!=(10,STATE_CHANNELS):
  raise ValueError(f'Blend alphas must have shape (10, {STATE_CHANNELS}), got {alphas.shape}.')
 return alphas
def validate_dap_weights(payload,config,expected_force_vars):
 if payload.get('calibration_type')!='damped_anomaly_persistence':
  raise ValueError('DAP weights calibration_type must be damped_anomaly_persistence.')
 if payload.get('fit_split')!='val':
  raise ValueError('DAP weights fit_split must be val.')
 expected_config_hash=current_config_sha256(config)
 if payload.get('config_sha256')!=expected_config_hash:
  raise ValueError(f'DAP config_sha256 does not match current config: {payload.get("config_sha256")} != {expected_config_hash}.')
 if tuple(payload.get('forcing_variables',()))!=tuple(expected_force_vars):
  raise ValueError(f'DAP forcing_variables={payload.get("forcing_variables")} does not match config {list(expected_force_vars)}.')
 expected_variables=('thetao','so','uo','vo','zos')
 if tuple(payload.get('variables',()))!=expected_variables:
  raise ValueError(f'DAP variables must be {expected_variables}.')
 gamma=payload.get('gamma')
 if not isinstance(gamma,dict):
  raise ValueError('DAP gamma must be an object keyed by variable.')
 for name in expected_variables:
  values=np.asarray(gamma.get(name),dtype=np.float32)
  if values.shape!=(10,):
   raise ValueError(f'DAP gamma for {name} must have shape (10,), got {values.shape}.')
 return {name:np.asarray(gamma[name],dtype=np.float32) for name in expected_variables}
