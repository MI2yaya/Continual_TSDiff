#!/usr/bin/env python3
"""
TSDiff Continual Learning Comparison Plots
Clean script with time series selection capability
"""

import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from pathlib import Path
from tqdm.auto import tqdm
import argparse

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.field_names import FieldName
from gluonts.transform import Identity

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.model.diffusion.sfdiff import SFDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    MaskInput,
    add_config_to_argparser
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GUIDANCE_MAP = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

class TSDiffPlotter:
    """Simple TSDiff forecast plotter with time series selection"""
    
    def __init__(self, config: dict, checkpoint_path: str,model_type: str):
        self.config = config
        self.device = config['device']
        self.model = self._load_model(checkpoint_path,model_type)
        
    def _load_model(self, checkpoint_path: str,model_type:str):
        """Load TSDiff model from checkpoint"""
        logger.info(f"Loading: {Path(checkpoint_path).name}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if model_type=="tsdiff":
            model = TSDiff(
                **getattr(diffusion_configs, self.config["diffusion_config"]),
                freq=self.config["freq"],
                use_features=self.config["use_features"],
                use_lags=self.config["use_lags"],
                normalization=self.config["normalization"],
                context_length=self.config["context_length"],
                prediction_length=self.config["prediction_length"],
                lr=self.config["lr"],
                init_skip=self.config["init_skip"],
                dropout_rate=self.config.get("dropout_rate", 0.01),
            )
        elif model_type=='sfdiff':
            model = SFDiff(
                **getattr(diffusion_configs, config["diffusion_config"]),
                freq=config["freq"],
                use_features=config["use_features"],
                normalization=config["normalization"],
                context_length=config["context_length"],
                prediction_length=config["prediction_length"],
                input_dim=config["input_dim"],
                lr=config["lr"],
                init_skip=config["init_skip"],
            )
        else:
            raise ValueError
        if "state_dict" in checkpoint:  # Lightning .ckpt
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:  # common torch .pth
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint  # raw state_dict saved directly

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading state_dict: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading state_dict: {unexpected}")

        model.to(self.device)
        model.eval()
        return model

    def generate_forecasts(self, dataset_name: str, start_index: int = 0, num_series: int = 1, num_samples: int = 100):
        """Generate forecasts for multiple time series starting from start_index."""
        from gluonts.dataset.common import ListDataset


        # --- Dataset Setup ---
        if dataset_name.startswith("custom"):
            from uncond_ts_diff.dataset import get_custom_dataset
            dataset = get_custom_dataset(
                dataset_name,
                samples=self.config.get("samples"),
                context_length=self.config.get("context_length"),
                prediction_length=self.config.get("prediction_length"),
                dt=self.config.get("dt"),
                q=self.config.get("q"),
            )
            freq = self.config.get("freq", "H")

            dataset_list = list(dataset)
            selected_data = dataset_list[start_index : start_index + num_series+1]

            # Convert to GluonTS ListDataset
            test = ListDataset(
                [
                    {
                        "start": (
                            d.get("start", "2020-01-01").to_timestamp(how="start")
                            if isinstance(d.get("start", "2020-01-01"), pd.Period)
                            else d.get("start", "2020-01-01")
                        ),
                        "target": d["target"],
                    }
                    for d in selected_data
                ],
                freq=freq,
            )

        else:
            # Built-in GluonTS dataset
            from gluonts.dataset.repository.datasets import get_dataset
            dataset = get_dataset(dataset_name, prediction_length=self.config["prediction_length"], regenerate=True)
            freq = str(dataset.metadata.freq)
            test = dataset.test

        self.config["freq"] = freq

        transformed_testdata = create_transforms(
            num_feat_dynamic_real=0, num_feat_static_cat=0, num_feat_static_real=0,
            time_features=self.model.time_features,
            prediction_length=self.config["prediction_length"]
        ).apply(test, is_train=False)

        test_splitter = create_splitter(
            past_length=self.config["context_length"] + max(self.model.lags_seq),
            future_length=self.config["prediction_length"],
            mode="test"
        )
        masking_transform = MaskInput(
            FieldName.TARGET, FieldName.OBSERVED_VALUES,
            self.config["context_length"], "none", 0
        )
        selected_data = list(transformed_testdata)[start_index:start_index + num_series]
        list_ds = selected_data
        
        transform = test_splitter + masking_transform



        # --- Build Predictor ---
        Guidance = GUIDANCE_MAP[self.config["sampler"]]
        sampler_kwargs = self.config.get("sampler_params", {})

        sampler = Guidance(
            model=self.model,
            prediction_length=self.config["prediction_length"],
            num_samples=num_samples,
            missing_scenario="none",
            missing_values=0,
            **sampler_kwargs,
        )

        predictor = sampler.get_predictor(
            input_transform=transform,
            batch_size=max(1, 1280 // num_samples),
            device=str(self.device),
        )

        # --- Forecasting ---
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=list_ds,
            predictor=predictor,
            num_samples=num_samples,
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        return forecasts, tss, freq

    def plot_forecast(self, forecast, ts, freq, method_name: str, ax):
        """Plot single forecast on given axis"""
        # Handle time series data
        historical_data = ts
        if isinstance(forecast, dict):
            forecast_start = forecast.get("start", "2020-01-01")
            if isinstance(forecast_start, pd.Period):
                forecast_start = forecast_start.to_timestamp(how='start')
        else:
            forecast_start = forecast.start_date
        
        # Convert historical index to Timestamp
        if hasattr(ts, 'index') and isinstance(ts.index, pd.PeriodIndex):
            historical_index = ts.index.to_timestamp(how='start')  # or 'end'
        else:
            historical_index = ts.index if hasattr(ts, 'index') else pd.date_range(
                end=pd.Timestamp(forecast_start) - pd.Timedelta(hours=1),
                periods=len(historical_data),
                freq=freq
            )

        # Convert forecast start to Timestamp if it's a Period
        if isinstance(forecast_start, pd.Period):
            forecast_start = forecast_start.to_timestamp(how='start')
        # Create forecast index
        forecast_index = pd.date_range(start=forecast_start, periods=len(forecast.mean), freq=freq)
        
        # Plot historical data (last portion)
        context_length = min(len(historical_data), self.config["context_length"])
        hist_start = max(0, len(historical_data) - context_length - 24)
        
        ax.plot(historical_index[hist_start:], historical_data[hist_start:], 
               'b-', linewidth=1.5, label='Ground Truth', alpha=0.8)
        
        # Plot forecast
        ax.plot(forecast_index, forecast.quantile(0.75), 'r-', linewidth=2, label='Median Forecast')
        
        # Plot prediction intervals
        if hasattr(forecast, 'quantile'):
            lower_90 = forecast.quantile(0.05)
            upper_90 = forecast.quantile(0.95)
            ax.fill_between(forecast_index, lower_90, upper_90, 
                           alpha=0.3, color='indianred', label='90% Interval')
            
            # lower_50 = forecast.quantile(0.25)
            # upper_50 = forecast.quantile(0.75)
            # ax.fill_between(forecast_index, lower_50, upper_50, 
            #                alpha=0.5, color='red', label='50% Interval')
        
        # Add forecast separator
        ax.axvline(x=forecast_start, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        
        # Formatting
        # ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
        ax.xaxis.set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper left')


def create_continual_learning_plots(config=None,start_series: int = 0, num_series: int = 1):
    """Create 5-method comparison plots with configurable series selection"""
    #!!!!!!!!!!!! CHANGE THE CKPT PATH TO YOURS!!!!
    checkpoints = { 
        # "Naive": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_naive/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
        # "Dropout": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_dropout/dropout_rate_0.5/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
        # "L1 Reg": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_score_l1/lambda_reg_2.0/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
        # "L2 Reg": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_score_l2/lambda_reg_2.0/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
        "Single Task": "C:/Users/micha/Downloads/Michael Petrizzo - Resume/Python/Continual_TSDiff/lightning_logs/version_26/best_checkpoint.ckpt",  #!!!!!!!!!!
    }
    target_dataset = config["dataset"]
    model_type = config['model_type'].lower()
    
    # Set seeds for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine figure layout based on num_series
    # Determine figure layout based on num_series
    num_methods = len(checkpoints)

    fig, axes = plt.subplots(
        num_series, num_methods,
        figsize=(5 * num_methods, 5 * num_series)
    )

    fig.suptitle(
        f'Continual Learning Performance (Series {start_series}-{start_series+num_series-1})'
        if num_series > 1 else f'Continual Learning Performance (Series {start_series})',
        fontsize=16, fontweight='bold'
    )

    # ✅ Ensure axes is always 2D
    if num_series == 1 and num_methods == 1:
        axes = np.array([[axes]])               # single cell → 2D
    elif num_series == 1:
        axes = axes[np.newaxis, :]              # 1 row, many cols
    elif num_methods == 1:
        axes = axes[:, np.newaxis]              # many rows, 1 col


    # Ensure axes is always 2D for consistency
    if num_series == 1:
        axes = np.atleast_2d(axes)
    
    logger.info(f"Generating plots for {num_series} time series starting from index {start_series}...")
    
    for method_idx, (method_name, checkpoint_path) in enumerate(checkpoints.items()):
        try:
            plotter = TSDiffPlotter(config, checkpoint_path,model_type)
            forecasts, tss, freq = plotter.generate_forecasts(
                target_dataset, start_index=start_series, num_series=num_series, num_samples=100
            )
            print(f"Len forecasts: {len(forecasts)}, Len tss: {len(tss)}")
            
            # Plot each time series
            for series_idx in range(len(forecasts)):
                if num_series == 1:
                    ax = axes[method_idx]
                    title_suffix = ""
                else:
                    ax = axes[series_idx, method_idx]
                    title_suffix = f" (TS{start_series + series_idx})"
                
                plotter.plot_forecast(forecasts[series_idx], tss[series_idx], freq, 
                                    method_name + title_suffix, ax)
            
            logger.info(f"✓ {method_name} plots completed")
            
        except Exception as e:
            logger.error(f"✗ {method_name} failed: {e}")
            # Handle error display for multiple series
            if num_series == 1:
                ax = axes[method_idx]
                ax.text(0.5, 0.5, f"Error\n{method_name}", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
            else:
                for series_idx in range(num_series):
                    ax = axes[series_idx, method_idx]
                    ax.text(0.5, 0.5, f"Error\n{method_name}", ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='red')
    
    plt.tight_layout()

    # Save plot
    if num_series == 1:
        output_file = f'continual_learning_comparison_series_{start_series}_{config["dataset"].replace(":","_")}.png'
    else:
        output_file = f'continual_learning_comparison_series_{start_series}_to_{start_series+num_series-1}_{config["dataset"].replace(":","_")}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Comparison plot saved: {output_file}")



def main(config=None):
    """Main function with configurable parameters"""
    logger.info("Starting TSDiff continual learning comparison")
    
    # ✅ CONFIGURE WHICH TIME SERIES TO PLOT
    
    # Option 1: Single time series at specific index
    # create_continual_learning_plots(start_series=0, num_series=1)  # First time series
    # create_continual_learning_plots(start_series=5, num_series=1)  # 6th time series
    
    
    # Option 2: Multiple time series (like original code)
    create_continual_learning_plots(config=config,start_series=1, num_series=3)  # First 3 time series
    # create_continual_learning_plots(start_series=10, num_series=3)  # Time series 10-12
    
    logger.info("Plotting completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "-m", "--model_type", type=str, required=True, help="Model Type (tsdiff or sfdiff)"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    main(config=config)
    
