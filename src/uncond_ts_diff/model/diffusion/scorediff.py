# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from uncond_ts_diff.arch.scorebackbones import ScoreBackboneModel
from uncond_ts_diff.model.diffusion._scorebase import ScoreDiffBase
from uncond_ts_diff.utils import extract 
from uncond_ts_diff.utils import make_diffusion_gif


class ScoreDiff(ScoreDiffBase):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        observation_dim,
        h_fn,
        R_inv,
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
        backbone_parameters['output_dim']=observation_dim
        print(backbone_parameters)
        
        self.backbone = ScoreBackboneModel(
            **backbone_parameters,
            init_skip=init_skip,
        )
        self.ema_rate = []  # [0.9999]
        self.ema_state_dicts = [
            copy.deepcopy(self.backbone.state_dict())
            for _ in range(len(self.ema_rate))
        ]
        self.observation_dim = observation_dim
        self.h_fn = h_fn
        self.R_inv = R_inv


    @torch.no_grad()
    def sample_n(
        self,
        y,
        num_samples: int = 1,
        cheap=True,
        base_strength=.1,
        plot=False,
        horizon = None,
    ):
        device = next(self.backbone.parameters()).device
        context_len = self.context_length
        full_len = context_len + self.prediction_length
        seq_len = full_len if horizon is None else context_len + horizon

        # initial noise
        samples = torch.randn((num_samples, seq_len, self.observation_dim), device=device)

        # scale y
        y, y_scale = self.scaler(y)
        y = y[:, :seq_len, :]

        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            # 1) clamp context to noised version at this timestep
            sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, samples.shape)[:, :context_len, :]
            sqrt_one_minus_t = extract(self.sqrt_one_minus_alphas_cumprod, t, samples.shape)[:, :context_len, :]
            noise_context = torch.randn_like(y[:, :context_len, :], device=device)
            x_context_t = sqrt_alpha_t * y[:, :context_len, :] + sqrt_one_minus_t * noise_context

            samples[:, :context_len, :] = x_context_t

            # 2) one reverse step
            samples = self.p_sample(
                samples,
                t,
                i,
                y,
                self.h_fn,
                self.R_inv,
                base_strength=base_strength,
                cheap=cheap,
                plot=plot,
            )

            # 3) re-clamp context after the step to avoid drift
            samples[:, :context_len, :] = x_context_t
        if plot:
            make_diffusion_gif()
        return samples * y_scale

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
