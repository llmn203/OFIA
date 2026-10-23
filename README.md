# OFIA

Research code for the paper:

> **Adapting Reanalysis-Trained Models to Forecast 3D Ocean States Using Near-Real-Time Products**
> (submitted to *Journal of Ocean Engineering and Science*)

OFIA adapts deep learning models pretrained on reanalysis data (Copernicus Marine multi-year reanalysis + ERA5) to near-real-time forecasting inputs (Copernicus Marine analysis and forecast product + GFS) in the Taiwan Strait. The pretrained backbone is frozen, and lightweight residual adapters (OHA, FSA, SIA) with a shared forecast-cycle context are trained on the near-real-time inputs.

In the code, OFIA appears under its internal development names (e.g. `dista_*`, `dual_interface_spatiotemporal_adapter`).

## Repository structure

```
models/       Backbone models (Temporal Transformer, ConvLSTM-UNet, ConvGRU-UNet, SimVP, Residual UNet, LSTM bottleneck, ...)
adapters/     OFIA adapters, ablation variants and generic parameter-efficient baselines
training/     Pretraining (hindcast), full fine-tuning and adapter training
evaluation/   Evaluation metrics, paired block bootstrap and evaluation scripts
environment/  Python dependencies and environment notes
```

## Requirements

Python 3.9 and PyTorch. Core dependencies are listed in `environment/requirements_core.txt`:

```
pip install -r environment/requirements_core.txt
```

## Data

Raw data are not included in this repository. They can be obtained from:

- Copernicus Marine Service: Global Ocean Physics Reanalysis and Global Ocean Physics Analysis and Forecast products (https://marine.copernicus.eu)
- ERA5 reanalysis: Copernicus Climate Data Store (https://cds.climate.copernicus.eu)
- GFS forecasts: NOAA/NCEP Global Forecast System

## Citation

If you use this code, please cite the paper (the citation will be added after publication).

## License

This project is released under the MIT License.