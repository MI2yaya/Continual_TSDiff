# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import math
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import Callback
from sklearn.linear_model import LinearRegression
from typing import Tuple
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
class ScoreStateMSECallback(Callback):
    def __init__(
            self,
            model,
            test_dataset,
            context_length,
            prediction_length,
            test_batch_size=32,
            num_mc_samples=5,
            eval_every=5,
            max_eval_batches=None,  # if set, only use first N batches
            fast_denoise=True,
            plot_samples=3,
        ):
            super().__init__()
            self.model = model
            self.test_dataset = test_dataset
            self.context_length = context_length
            self.prediction_length = prediction_length
            self.test_batch_size = test_batch_size
            self.num_mc_samples = num_mc_samples
            self.eval_every = eval_every
            self.max_eval_batches = max_eval_batches
            self.fast_denoise = fast_denoise
            self.plot_samples=plot_samples
            self.test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)



    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module=None):
        if (trainer.current_epoch + 1) % self.eval_every != 0:
            return

        device = next(self.model.backbone.parameters()).device
        mse_context_total = 0.0
        mse_future_total = 0.0
        count = 0

        self.model.eval()

        for batch_idx, batch in enumerate(self.test_loader):
            if self.max_eval_batches is not None and batch_idx >= self.max_eval_batches:
                break

            past_state = batch["past_state"].to(device)
            future_state = batch["future_state"].to(device)
            past_observed = batch["past_observation"].to(device)
            future_observed = torch.zeros(
                (past_observed.size(0), self.prediction_length, past_observed.size(2)),
                device=device
            )

            y = torch.cat([past_observed, future_observed], dim=1).to(dtype=torch.float32)

            y_repeat = y.unsqueeze(0).repeat(self.num_mc_samples, 1, 1, 1)
            B, S, D = y_repeat.shape[1], y_repeat.shape[2], y_repeat.shape[3] if len(y_repeat.shape) == 4 else 1
            y_repeat = y_repeat.view(-1, S, D)  # flatten batch dimension

            if self.fast_denoise:
                generated = self.model.fast_sample(y_repeat,num_steps=50)
            else:
                generated = self.model.sample_n(y_repeat, num_samples=y_repeat.size(0),cheap=True,base_strength=.5)

            # Reshape back to [num_mc_samples, batch, seq, dim]
            generated = generated.view(self.num_mc_samples, B, S, D)
            generated_mean = generated.mean(dim=0)

            # Split context / future
            pred_context = generated_mean[:, :self.context_length]
            pred_future = generated_mean[:, self.context_length:]

            mse_context_total += ((pred_context - past_state) ** 2).sum()
            mse_future_total += ((pred_future - future_state) ** 2).sum()
            count += past_state.numel()


        mse_context_total /= count
        mse_future_total /= count

        trainer.logger.log_metrics({
            "mse_context": mse_context_total,
            "mse_future": mse_future_total,
        }, step=trainer.current_epoch)

        self.model.train()



def compute_metrics(pred, true):
    
    mean_pred = pred.mean(axis=0)
    mse = np.mean((mean_pred - true) ** 2)
    rmse = np.sqrt(mse)
    nd = np.sum(np.abs(mean_pred - true)) / (np.sum(np.abs(true)) + 1e-8)
    nrmse = rmse / (np.std(true) + 1e-8)
    return {"ND": nd, "NRMSE": nrmse, "CRPS": mse}