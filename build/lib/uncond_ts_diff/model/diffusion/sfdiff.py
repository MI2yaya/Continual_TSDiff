# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from uncond_ts_diff.arch import BackboneModel
from uncond_ts_diff.model.diffusion._sfbase import SFDiffBase



class SFDiff(SFDiffBase):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        normalization="none",
        init_skip=True,
        lr=1e-3,
        dropout_rate=0.01,
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

        print(f'Backbone params: {backbone_parameters}')

        self.state_dim = backbone_parameters["state_dim"]
        self.meas_dim = backbone_parameters["measurement_dim"]
        self.observation_dim
        self.backbone = BackboneModel(
            **backbone_parameters,
            init_skip=init_skip,
        )
        self.ema_rate = []  # [0.9999]
        self.ema_state_dicts = [
            copy.deepcopy(self.backbone.state_dict())
            for _ in range(len(self.ema_rate))
        ]

    def _extract_features(self, data):
        prior_state = data["past_state"][:, -self.context_length, :] # B, L, D
        s_prior_state, state_scale = self.scaler(prior_state)
        s_future_state = data["future_state"] / state_scale
        s_whole_state = torch.cat([s_prior_state, s_future_state], dim=1)

        prior_obs = data["past_observation"][:, -self.context_length, :] #B, L, D
        s_prior_obs, meas_scale = self.scaler(prior_obs)
        s_future_obs = data['future_observation'] / meas_scale
        s_whole_obs = torch.cat([s_prior_obs,s_future_obs], dim=1)

        print(f"State Shape: {s_whole_state.shape}\nState to 10:{s_whole_state[:10]}")
        print(f"Obs Shape: {s_whole_obs.shape}\nMeas to 10:{s_whole_obs[:10]}")

        return s_whole_state, s_whole_obs, state_scale, meas_scale

    @torch.no_grad()
    def sample_n(
        self,
        num_samples: int = 1,
        return_lags: bool = False,
    ):
        device = next(self.backbone.parameters()).device
        seq_len = self.context_length + self.prediction_length

        samples = torch.randn(
            (num_samples, seq_len, self.input_dim), device=device
        )

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            samples = self.p_sample(samples, t, i, features=None)

        samples = samples.cpu().numpy()

        if return_lags:
            return samples

        return samples[..., 0]

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
