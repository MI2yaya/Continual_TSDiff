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

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.field_names import FieldName

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    MaskInput,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GUIDANCE_MAP = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

class TSDiffPlotter:
    """Simple TSDiff forecast plotter with time series selection"""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = torch.device(config.get("device", "cuda:1"))
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path: str) -> TSDiff:
        """Load TSDiff model from checkpoint"""
        logger.info(f"Loading: {Path(checkpoint_path).name}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def generate_forecasts(self, dataset_name: str, start_index: int = 0, num_series: int = 1, num_samples: int = 100):
        """Generate forecasts for multiple time series starting from start_index"""
        # Load dataset
        dataset = get_dataset(dataset_name, prediction_length=24, regenerate=True)
        
        # Auto-correct frequency
        actual_freq = str(dataset.metadata.freq)
        if actual_freq != self.config.get("freq", "H"):
            self.config["freq"] = actual_freq
        
        # Setup transformation and sampler
        transformation = create_transforms(
            num_feat_dynamic_real=0, num_feat_static_cat=0, num_feat_static_real=0,
            time_features=self.model.time_features,
            prediction_length=self.config["prediction_length"],
        )
        
        Guidance = GUIDANCE_MAP[self.config["sampler"]]
        sampler_kwargs = self.config.get("sampler_params", {})
        sampler = Guidance(
            model=self.model, prediction_length=self.config["prediction_length"],
            num_samples=num_samples, missing_scenario="none", missing_values=0,
            **sampler_kwargs,
        )
        
        # Setup data pipeline
        transformed_testdata = transformation.apply(dataset.test, is_train=False)
        test_splitter = create_splitter(
            past_length=self.config["context_length"] + max(self.model.lags_seq),
            future_length=self.config["prediction_length"], mode="test",
        )
        masking_transform = MaskInput(
            FieldName.TARGET, FieldName.OBSERVED_VALUES,
            self.config["context_length"], "none", 0,
        )
        test_transform = test_splitter + masking_transform
        
        # Create predictor
        predictor = sampler.get_predictor(
            test_transform, batch_size=max(1, 1280 // num_samples), device=str(self.device),
        )
        
        # ✅ SELECT SPECIFIC TIME SERIES RANGE
        selected_series = []
        try:
            # Convert to list and slice the desired range
            testdata_list = list(transformed_testdata)
            end_index = min(start_index + num_series, len(testdata_list))
            
            if start_index >= len(testdata_list):
                logger.warning(f"Start index {start_index} >= dataset size {len(testdata_list)}. Using index 0.")
                start_index = 0
                end_index = min(num_series, len(testdata_list))
            
            selected_series = testdata_list[start_index:end_index]
            
        except MemoryError:
            # For large datasets, use itertools
            logger.info(f"Large dataset detected, using itertools for indices {start_index}-{start_index+num_series-1}")
            selected_series = list(itertools.islice(transformed_testdata, start_index, start_index + num_series))
        
        if not selected_series:
            logger.warning(f"No series found at indices {start_index}-{start_index+num_series-1}. Using first series.")
            selected_series = [next(iter(transformed_testdata))]
        
        logger.info(f"Selected {len(selected_series)} time series starting from index {start_index}")
        
        # Generate forecasts for selected series
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=selected_series, predictor=predictor, num_samples=num_samples,
        )
        
        forecasts = list(tqdm(forecast_it, total=len(selected_series), desc="Generating forecasts"))
        tss = list(ts_it)
        
        return forecasts, tss, dataset.metadata.freq

    def plot_forecast(self, forecast, ts, freq, method_name: str, ax):
        """Plot single forecast on given axis"""
        # Handle time series data
        historical_data = ts
        forecast_start = forecast.start_date
        
        # Handle PeriodIndex
        if hasattr(ts, 'index') and isinstance(ts.index, pd.PeriodIndex):
            historical_index = ts.index.to_timestamp()
        else:
            historical_index = ts.index if hasattr(ts, 'index') else pd.date_range(
                end=forecast_start - pd.Timedelta(hours=1), periods=len(historical_data), freq=freq
            )
        
        # Handle forecast start
        if hasattr(forecast_start, 'to_timestamp'):
            forecast_start = forecast_start.to_timestamp()
        
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


def create_continual_learning_plots(start_series: int = 0, num_series: int = 1):
    """Create 5-method comparison plots with configurable series selection"""
    #!!!!!!!!!!!! CHANGE THE CKPT PATH TO YOURS!!!!
    checkpoints = { 
        # "Naive": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_naive/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
        # "Dropout": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_dropout/dropout_rate_0.5/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
        # "L1 Reg": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_score_l1/lambda_reg_2.0/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
        # "L2 Reg": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_score_l2/lambda_reg_2.0/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
        "Single Task": "/Users/monicastudent/Downloads/Continual_TSDiff/lightning_logs/version_3/checkpoints/m4_hourly-epoch=014-train_loss=0.059.ckpt" 
    }
    # checkpoints = {
    #     "Naive": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_naive/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
    #     "Dropout": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_dropout/dropout_rate_0.3/task_3_uber_tlc_hourly/uber_tlc_hourly_checkpoint_best.pth",
    #     "L1 Reg": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_score_l1/lambda_reg_2.0/task_1_kdd_cup_2018_without_missing/kdd_cup_2018_without_missing_checkpoint_best.pth",
    #     "L2 Reg": "/export/home/anandr/diffusion/Continual_TSDiff/full_experiments_3/order_1_kdd_cup_pedestrian_counts_uber_tlc/method_score_l2/lambda_reg_2.0/task_1_kdd_cup_2018_without_missing/kdd_cup_2018_without_missing_checkpoint_best.pth",
    #     "Single Task": "/export/home/anandr/diffusion/Continual_TSDiff/01_kdd_cup/kdd_cup_2018_without_missing_checkpoint_best.pth"
    # }
    
    config = yaml.safe_load(open("configs/eval_continual.yaml"))
    target_dataset = "m4_hourly"                #!!!!!!!!!! CHANGE THIS TOO!!!!
    
    # Set seeds for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine figure layout based on num_series
    if num_series == 1:
        # Single row of methods
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'Continual Learning Performance on KDD Cup Dataset (Series {start_series})', 
                    fontsize=16, fontweight='bold')
    else:
        # Multiple rows: one row per time series
        fig, axes = plt.subplots(num_series, 5, figsize=(25, 5 * num_series))
        fig.suptitle(f'Continual Learning Performance on KDD Cup Dataset (Series {start_series}-{start_series+num_series-1})', 
                    fontsize=16, fontweight='bold')
        if num_series == 1:
            axes = axes.reshape(1, -1)  # Ensure 2D array
    
    logger.info(f"Generating plots for {num_series} time series starting from index {start_series}...")
    
    for method_idx, (method_name, checkpoint_path) in enumerate(checkpoints.items()):
        try:
            plotter = TSDiffPlotter(config, checkpoint_path)
            forecasts, tss, freq = plotter.generate_forecasts(
                target_dataset, start_index=start_series, num_series=num_series, num_samples=100
            )
            
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
        output_file = f'continual_learning_comparison_series_{start_series}.png'
    else:
        output_file = f'continual_learning_comparison_series_{start_series}_to_{start_series+num_series-1}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Comparison plot saved: {output_file}")


def main():
    """Main function with configurable parameters"""
    logger.info("Starting TSDiff continual learning comparison")
    
    # ✅ CONFIGURE WHICH TIME SERIES TO PLOT
    
    # Option 1: Single time series at specific index
    # create_continual_learning_plots(start_series=0, num_series=1)  # First time series
    # create_continual_learning_plots(start_series=5, num_series=1)  # 6th time series
    
    # Option 2: Multiple time series (like original code)
    create_continual_learning_plots(start_series=1, num_series=1)  # First 3 time series
    # create_continual_learning_plots(start_series=10, num_series=3)  # Time series 10-12
    
    logger.info("Plotting completed successfully!")


if __name__ == "__main__":
    main()
