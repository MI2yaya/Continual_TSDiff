# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from uncond_ts_diff.arch.sfbackbones import SFBackboneModel
from uncond_ts_diff.model.diffusion._sfbase import SFDiffBase



class SFDiff(SFDiffBase):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        state_dim,
        observation_dim,
        normalization="none",
        init_skip=True,
        lr=1e-3,
        dropout_rate=0.01
    ):
        super().__init__(
            backbone_parameters,
            timesteps=timesteps,
            diffusion_scheduler=diffusion_scheduler,
            context_length=context_length,
            prediction_length=prediction_length,
            normalization=normalization,
            lr=lr,
            dropout_rate=dropout_rate,
        )
        self.lags_seq=[0] #just for callback past_length=self.context_length + max(self.model.lags_seq),
        backbone_parameters["dropout"] = dropout_rate
        backbone_parameters["observation_dim"] = observation_dim
        backbone_parameters['output_dim']=state_dim
        print(backbone_parameters)
        
        self.backbone = SFBackboneModel(
            **backbone_parameters,
            init_skip=init_skip,
        )
        self.ema_rate = []  # [0.9999]
        self.ema_state_dicts = [
            copy.deepcopy(self.backbone.state_dict())
            for _ in range(len(self.ema_rate))
        ]
        self.state_dim = state_dim

    @torch.no_grad()
    def sample_n(
        self,
        num_samples: int = 1,
        features = None
    ):
        device = next(self.backbone.parameters()).device
        seq_len = self.context_length  + self.prediction_length #true seq len

        samples = torch.randn(
            (num_samples, seq_len, self.state_dim), device=device
        )

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            samples = self.p_sample(samples, t, i, features=features)
        return samples

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for rate, state_dict in zip(self.ema_rate, self.ema_state_dicts):
            update_ema(state_dict, self.backbone.state_dict(), rate=rate)


def update_ema(target_state_dict, source_state_dict, rate=0.99):
    with torch.no_grad():
        for key, value in source_state_dict.items():
            ema_value = target_state_dict[key]
            ema_value.copy_(
                rate * ema_value + (1.0 - rate) * value.cpu(),
                non_blocking=True,
            )
