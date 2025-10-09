# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import math
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from pytorch_lightning import Callback
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple

def linear_pred_score(
    samples: np.ndarray,
    context_length: int,
    prediction_length: int,
    test_dataset: np.ndarray,
    scaling_type: str = "mean",
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Compute a simple linear predictive score using scikit-learn LinearRegression.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape [N, T], synthetic or real training samples.
    context_length : int
        Number of past time steps used for regression input.
    prediction_length : int
        Number of steps ahead to forecast.
    test_dataset : np.ndarray
        Array of test samples of shape [M, T].
    scaling_type : str
        Scaling method: "mean" or "min-max". Default "mean".

    Returns
    -------
    metrics : dict
        Dictionary of ND and NRMSE scores.
    y_true : np.ndarray
        True test targets of shape [M, prediction_length].
    y_pred : np.ndarray
        Predicted forecasts of shape [M, prediction_length].
    """
    assert samples.shape[1] >= context_length + prediction_length
    print(test_dataset.shape)
    assert test_dataset.shape[1] >= context_length + prediction_length

    # Split samples into training X and Y
    X_train = samples[:, :context_length]
    Y_train = samples[:, context_length:context_length + prediction_length]

    # Train simple linear regression
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Prepare test data
    X_test = test_dataset[:, :context_length]
    Y_true = test_dataset[:, context_length:context_length + prediction_length]

    # Predict
    Y_pred = model.predict(X_test)

    # Compute metrics
    mean_pred = Y_pred.mean(axis=0)
    mse = np.mean((Y_pred - Y_true) ** 2)
    rmse = np.sqrt(mse)
    nd = np.sum(np.abs(Y_pred - Y_true)) / (np.sum(np.abs(Y_true)) + 1e-8)
    nrmse = rmse / (np.std(Y_true) + 1e-8)

    metrics = {"ND": nd, "NRMSE": nrmse}
    return metrics, Y_true, Y_pred

class GradNormCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_before_optimizer_step(
        self,
        trainer,
        pl_module,
        optimizer,
        opt_idx: int,
    ) -> None:
        return pl_module.log(
            "grad_norm", self.grad_norm(pl_module.parameters()), prog_bar=True
        )

    def grad_norm(self, parameters):
        parameters = [p for p in parameters if p.grad is not None]
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), 2).to(device) for p in parameters]
            ),
            2,
        )
        return total_norm


class SFPredictiveScoreCallback(Callback):
    def __init__(
        self,
        context_length,
        prediction_length,
        model,
        train_dataloader,
        train_batch_size,
        test_dataset,
        eval_every=10,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model = model
        self.train_dataloader = train_dataloader
        self.train_batch_size = train_batch_size
        self.test_dataset = test_dataset
        self.eval_every = eval_every
        # Number of samples used to train the downstream predictor
        self.n_pred_samples = 10000

    def _generate_real_samples(
        self,
        data_loader,
        num_samples: int,
        n_timesteps: int,
        batch_size: int,
        cache_path: Path,
    ):
        if cache_path.exists():
            real_samples = np.load(cache_path)
            if len(real_samples) == num_samples:
                return real_samples

        real_samples = []
        data_iter = iter(data_loader)
        n_iters = math.ceil(num_samples / batch_size)
        for i in range(n_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)
            ts = np.concatenate(
                [batch["past_state"], batch["future_state"]], axis=1 #eerrrrr is this right??
            )[:, -n_timesteps:]
            real_samples.append(ts)

        real_samples = np.concatenate(real_samples, axis=0)[:num_samples]
        np.save(cache_path, real_samples)

        return real_samples

    def _generate_synth_samples(self, model, num_samples: int, batch_size: int = 1000):
        synth_samples = []

        n_iters = math.ceil(num_samples / batch_size)
        for i in range(n_iters):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            samples = model.sample_n(num_samples=current_batch_size)
            synth_samples.append(samples)

        synth_samples = np.concatenate(synth_samples, axis=0)[:num_samples]
        print(synth_samples.shape)
        return synth_samples

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.eval_every == 0:
            device = next(pl_module.backbone.parameters()).device
            pl_module.eval()
            assert pl_module.training is False

            real_samples = self._generate_real_samples(
                self.train_dataloader,
                self.n_pred_samples,
                self.context_length + self.prediction_length,
                self.train_batch_size,
                cache_path=Path(trainer.logger.log_dir) / "real_samples.npy",
            )
            synth_samples = self._generate_synth_samples(
                self.model,
                self.n_pred_samples,
                self.train_batch_size
            )

            # Train using synthetic samples, test on test set
            synth_metrics, _, _ = linear_pred_score(
                synth_samples,
                self.context_length,
                self.prediction_length,
                self.test_dataset,
                scaling_type="mean",
            )

            # Train using real samples, test on test set
            scaled_real_samples, _ = self.model.scaler(
                torch.from_numpy(real_samples).to(device),
                torch.from_numpy(np.ones_like(real_samples)).to(device),
            )
            real_metrics, _, _ = linear_pred_score(
                scaled_real_samples.cpu().numpy(),
                self.context_length,
                self.prediction_length,
                self.test_dataset,
                scaling_type="mean",
            )

            pl_module.log_dict(
                {
                    "synth_linear_ND": synth_metrics["ND"],
                    "synth_linear_NRMSE": synth_metrics["NRMSE"],
                    "real_linear_ND": real_metrics["ND"],
                    "real_linear_NRMSE": real_metrics["NRMSE"],
                }
            )

            pl_module.train()

import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
class StateMSECallback(Callback):
    def __init__(
            self,
            context_length,
            prediction_length,
            model,
            test_dataset,
            test_batch_size=32,
            eval_every=10,
        ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model = model
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size
        self.eval_every = eval_every
        self.test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)



    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module=None):
                # Only evaluate every `eval_every` epochs
        if (trainer.current_epoch + 1) % self.eval_every != 0:
            return

        device = next(self.model.backbone.parameters()).device
        mse_context_total = 0.0
        mse_future_total = 0.0
        count = 0

        for batch in self.test_loader:
            past_state = batch["past_state"].to(device, dtype=torch.float32)
            future_state = batch["future_state"].to(device, dtype=torch.float32)
            
            past_observed = batch["past_observation"]
            future_observed = torch.zeros((past_observed.shape[0], self.prediction_length, past_observed.shape[2]))
            features = torch.cat([past_observed, future_observed], dim=1).to(device=device, dtype=torch.float32)

            batch_size = past_state.shape[0]

            # Generate predictions
            generated = self.model.sample_n(num_samples=batch_size, features=features)
            generated = torch.tensor(generated, device=device, dtype=torch.float32)

            # Compute MSE
            mse_context = ((generated[:, :self.context_length, :] - past_state) ** 2).mean()
            mse_future = ((generated[:, self.context_length:, :] - future_state) ** 2).mean()

            mse_context_total += mse_context.item() * batch_size
            mse_future_total += mse_future.item() * batch_size
            count += batch_size

        mse_context_total /= count
        mse_future_total /= count

        trainer.logger.log_metrics({
            "mse_context": mse_context_total,
            "mse_future": mse_future_total
        }, step=trainer.current_epoch)



def compute_metrics(pred, true):
    mean_pred = pred.mean(axis=0)
    mse = np.mean((mean_pred - true) ** 2)
    rmse = np.sqrt(mse)
    nd = np.sum(np.abs(mean_pred - true)) / (np.sum(np.abs(true)) + 1e-8)
    nrmse = rmse / (np.std(true) + 1e-8)
    return {"ND": nd, "NRMSE": nrmse, "CRPS": mse}