#!/usr/bin/env python3
"""
TSDiff Continual Learning Comparison Plots (State Forecast)
"""

import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import uncond_ts_diff.configs as diffusion_configs

from uncond_ts_diff.model.diffusion.sfdiff import SFDiff
from uncond_ts_diff.dataset import get_custom_dataset

from uncond_ts_diff.utils import (
    train_test_val_splitter,
    time_splitter,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateForecastPlotter:
    """Generates state forecasts from SFDiff models and plots them."""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = config['device']
        self.model = self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str):
        logger.info(f"Loading model checkpoint: {Path(checkpoint_path).name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        scaling = int(self.config['dt']**-1)
        context_length = self.config["context_length"] * scaling
        prediction_length = self.config["prediction_length"] * scaling

        model = SFDiff(
            **getattr(diffusion_configs, self.config["diffusion_config"]),
            state_dim=self.config["state_dim"],
            observation_dim=self.config["observation_dim"],
            normalization=self.config["normalization"],
            context_length=context_length,
            prediction_length=prediction_length,
            lr=self.config["lr"],
            init_skip=self.config["init_skip"]
        )

        # Load state_dict
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys in state_dict: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in state_dict: {unexpected}")

        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def generate_state_forecasts(self, dataset_name: str, start_index=0, num_series=1, num_samples=100):
        """Generate forecasts for multiple time series."""
        scaling = int(self.config['dt']**-1)

        dataset = get_custom_dataset(
            dataset_name,
            samples=self.config['data_samples'],
            context_length=self.config["context_length"],
            prediction_length=self.config["prediction_length"],
            dt=self.config['dt'],
            q=self.config['q']
        )

        selected_series = dataset[start_index:start_index + num_series]
        time_series = time_splitter(selected_series, self.config["context_length"] * scaling, self.config["prediction_length"] * scaling)
        forecasts = []
        for series in time_series:

            past_observation = torch.as_tensor(series["past_observation"], dtype=torch.float32)
            if past_observation.ndim == 2:  # shape (batch, seq_len)
                past_observation = past_observation.unsqueeze(0) 
            #future_observed = torch.zeros((past_observation.shape[0], self.config["prediction_length"] * scaling, past_observation.shape[2]))
            future_observed = torch.as_tensor(series['future_observation'],dtype=torch.float32)
            if future_observed.ndim == 2:  # shape (batch, seq_len)
                future_observed = future_observed.unsqueeze(0) 
            features = torch.cat([past_observation, future_observed], dim=1).to(device=self.model.device, dtype=torch.float32)


            # Generate samples from model
            generated = self.model.sample_n(num_samples=1, features=features)  # shape: (1, total_length, state_dim)
            forecasts.append(generated[0])

        return forecasts, time_series

    def plot_forecast(self, forecast, series_data, ax, title="Forecast"):
        """Plot a single forecast against ground truth states."""
        context_len = forecast.shape[0] - (self.config["prediction_length"] * int(self.config['dt']**-1))
        past_state = series_data["past_state"]
        future_state = series_data["future_state"]
        
        total_state = np.concatenate([past_state, future_state], axis=0)
        
        past_observation = series_data['past_observation']
        future_observation = series_data['future_observation']
        total_obs = np.concatenate([past_observation, future_observation], axis=0)

        ax.plot(total_state[:, 0], 'b-', linewidth=1.5, label="Ground Truth")
        ax.plot(total_obs[:, 0], 'g-', linewidth=1.5, label="Observation")
        ax.plot(forecast[:, 0], 'r--', linewidth=2, label="State Forecast")
        ax.axvline(x=len(past_state)-1, color='gray', linestyle='--', alpha=0.7)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(fontsize=10)

def create_continual_learning_plots(config, start_series=0, num_series=1):
    """Create plots for multiple time series."""
    checkpoints = {
        "Single Task": config["checkpoint_path"]
    }

    num_methods = len(checkpoints)
    fig, axes = plt.subplots(num_series, num_methods, figsize=(5*num_methods, 4*num_series))

    # Ensure axes is always 2D
    if num_series == 1 and num_methods == 1:
        axes = np.array([[axes]])
    elif num_series == 1:
        axes = axes[np.newaxis, :]           # 1 row, multiple columns
    elif num_methods == 1:
        axes = axes[:, np.newaxis]           # multiple rows, 1 column

    for method_idx, (method_name, ckpt_path) in enumerate(checkpoints.items()):
        #try:
            plotter = StateForecastPlotter(config, ckpt_path)
            forecasts, series_list = plotter.generate_state_forecasts(
                config["dataset"], start_index=start_series, num_series=num_series, num_samples=100
            )

            for series_idx, forecast in enumerate(forecasts):
                ax = axes[series_idx, method_idx]
                plotter.plot_forecast(forecast, series_list[series_idx], ax, title=f"{method_name} (TS{start_series + series_idx})")
            
            logger.info(f"✓ {method_name} plots completed")

            '''
        except Exception as e:
            logger.error(f"✗ {method_name} failed: {e}")
            for series_idx in range(num_series):
                ax = axes[series_idx, method_idx]
                ax.text(0.5, 0.5, f"Error\n{method_name}", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        '''
    plt.tight_layout()
    plt.savefig(f"continual_learning_states_{start_series}_to_{start_series+num_series-1}_{config['dataset'].replace(':','_')}.png", dpi=300)
    plt.close(fig)
    logger.info("Comparison plots saved.")

def main(config_path):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    create_continual_learning_plots(config, start_series=0, num_series=3)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    args = parser.parse_args()

    main(args.config)
