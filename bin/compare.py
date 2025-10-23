#!/usr/bin/env python3
"""
DDPM vs Kalman Filter Comparison Script

Compares DDPM forecasts against a classic Kalman Filter baseline
on a selected dataset. Computes MSE (context + future) and plots
median forecasts for both models.
"""

import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model.diffusion.scorediff import ScoreDiff
from uncond_ts_diff.dataset import get_custom_dataset
from uncond_ts_diff.utils import time_splitter
from dataGeneration import make_kf_matrices_for_sinusoid

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------
#   Utility functions
# -------------------------------

def mse(a, b):
    """Compute Mean Squared Error."""
    return np.mean((a - b) ** 2)


# -------------------------------
#   Kalman Filter Baseline
# -------------------------------
class SimpleKalmanFilter:
    """
    Minimal Kalman Filter baseline for comparison.
    Uses system matrices from dataset generator (A, H, Q, R).
    """

    def __init__(self, A, H, Q, R):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

    def forward(self, y, x0=None, P0=None):
        """Run Kalman filter through the given observations y."""
        T = y.shape[0]
        n = self.A.shape[0]
        xs = np.zeros((T, n))

        if x0 is None:
            x = np.zeros((n,))
        else:
            x = x0

        if P0 is None:
            P = np.eye(n)
        else:
            P = P0

        for t in range(T):
            # Predict
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            # Update
            K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
            x = x_pred + K @ (y[t] - self.H @ x_pred)
            P = (np.eye(n) - K @ self.H) @ P_pred

            xs[t] = x

        return xs

class DDPMForecaster:
    """Wrapper to load model + generate forecasts."""

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = config["device"]
        self.checkpoint_path = checkpoint_path
        self.model = None

    def _load_model(self, h_fn, R_inv):
        logger.info(f"Loading model checkpoint: {Path(self.checkpoint_path).name}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        scaling = int(self.config["dt"] ** -1)
        context_length = self.config["context_length"] * scaling
        prediction_length = self.config["prediction_length"] * scaling

        model = ScoreDiff(
            **getattr(diffusion_configs, self.config["diffusion_config"]),
            observation_dim=self.config["observation_dim"],
            normalization=self.config["normalization"],
            context_length=context_length,
            prediction_length=prediction_length,
            lr=self.config["lr"],
            init_skip=self.config["init_skip"],
            h_fn=h_fn,
            R_inv=R_inv,
        )

        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        self.model = model


# -------------------------------
#   Comparison Function
# -------------------------------
def compare_models(config, num_series=10, num_runs=1, plot=False):
    scaling = int(config["dt"] ** -1)

    # Prepare dataset + model
    dataset, generator = get_custom_dataset(
        config["dataset"],
        samples=config["data_samples"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        dt=config["dt"],
        q=config["q"],
        r=config["r"],
        observation_dim=config["observation_dim"],
    )

    ddpm_forecaster = DDPMForecaster(config, config["checkpoint_path"])
    ddpm_forecaster._load_model(generator.h_fn, generator.R_inv)

    # Kalman setup


    mse_results = {
        "DDPM": {"context": [], "future": []},
        "KF": {"context": [], "future": []},
    }
    example = None

    for s_idx in range(num_series):
        logger.info(f"--- Series {s_idx + 1}/{num_series} ---")

        # Pull one time series
        series = dataset[s_idx]
        time_series = time_splitter(
            [series],
            config["context_length"] * scaling,
            config["prediction_length"] * scaling,
        )[0]

        y_full = np.concatenate([time_series["past_observation"], time_series["future_observation"]], axis=0)
        true_states = np.concatenate([time_series["past_state"], time_series["future_state"]], axis=0)[:, 0]

        context_len = config["context_length"] * scaling
        future_len = config["prediction_length"] * scaling
        ddpm_forecasts = []
        kf_forecasts = []
        for run_idx in range(num_runs):
            logger.info(f"--- Sub-Run {run_idx + 1}/{num_runs} ---")
            # --- DDPM forecast ---
            past_obs = torch.as_tensor(time_series["past_observation"], dtype=torch.float32).unsqueeze(0)
            A, H, Q, R = make_kf_matrices_for_sinusoid(generator, past_obs=past_obs,mode="osc")
            kalman = SimpleKalmanFilter(A, H, Q, R)
            
            future_obs = torch.zeros((1, future_len, past_obs.shape[2]))
            y = torch.cat([past_obs, future_obs], dim=1).to(ddpm_forecaster.device)

            samples = ddpm_forecaster.model.sample_n(y, num_samples=100, cheap=False, base_strength=0.5)
            forecast_vals = samples.cpu().numpy()[:, :, 0]
            median_forecast = np.median(forecast_vals, axis=0)
            ddpm_forecasts.append(median_forecast)

            # --- Kalman forecast ---
            kf_states = kalman.forward(y_full.squeeze())
            kf_forecasts.append(kf_states[:, 0])

            # --- Compute MSEs ---
            ddpm_ctx_mse = mse(median_forecast[:context_len], true_states[:context_len])
            ddpm_fut_mse = mse(median_forecast[context_len:], true_states[context_len:])
            kf_ctx_mse = mse(kf_states[:context_len, 0], true_states[:context_len])
            kf_fut_mse = mse(kf_states[context_len:, 0], true_states[context_len:])

            mse_results["DDPM"]["context"].append(ddpm_ctx_mse)
            mse_results["DDPM"]["future"].append(ddpm_fut_mse)
            mse_results["KF"]["context"].append(kf_ctx_mse)
            mse_results["KF"]["future"].append(kf_fut_mse)
        if example is None:
            example = {
                "true_states": true_states,
                "ddpm_forecast": np.median(np.stack(ddpm_forecasts), axis=0),
                "kf_forecast": np.median(np.stack(kf_forecasts), axis=0),
                "context_len": context_len,
                "future_len": future_len,
            }

    # --- Aggregate & Print ---
    logger.info("\n=== Aggregated MSE Results (mean ± std) ===")
    for model in ["DDPM", "KF"]:
        ctx_mean, ctx_std = np.mean(mse_results[model]["context"]), np.std(mse_results[model]["context"])
        fut_mean, fut_std = np.mean(mse_results[model]["future"]), np.std(mse_results[model]["future"])
        logger.info(f"{model}: Context MSE={ctx_mean:.4f} ± {ctx_std:.4f}, "
                    f"Future MSE={fut_mean:.4f} ± {fut_std:.4f}")

    if plot and example is not None:
        plt.figure(figsize=(10, 5))
        t = np.arange(example["true_states"].shape[0])
        ctx = example["context_len"]

        plt.plot(t, example["true_states"], label="True State", color="black", linewidth=2)
        plt.plot(t, example["ddpm_forecast"], label="DDPM Forecast", color="tab:blue")
        plt.plot(t, example["kf_forecast"], label="Kalman Filter", color="tab:orange", linestyle="--")

        plt.axvline(ctx, color="gray", linestyle=":", label="Forecast Start")
        plt.title("DDPM vs Kalman Filter Forecast Comparison")
        plt.xlabel("Time Step")
        plt.ylabel("State")
        plt.legend()
        plt.tight_layout()
        plt.show()


    return mse_results



def main(config_path):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    compare_models(config,plot=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
