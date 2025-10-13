# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
import math

from uncond_ts_diff.utils import extract


class NOPScaler(torch.nn.Module):

    def __init__(self, dim=1, keepdim=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, observed_mask=None, stats=None):
        scale = torch.ones(x.shape[0], x.shape[2], device=x.device)
        if self.keepdim:
            scale = scale.unsqueeze(1)
        return x, scale
    
class MeanScaler(torch.nn.Module):
    """
    Scale by the mean absolute value of the time series.
    Similar to GluonTS MeanScaler.
    """
    def __init__(self, dim=1, keepdim=True, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.eps = eps

    def forward(self, x, observed_mask=None, stats=None):
        """
        Args:
            x: [batch, seq_len, features]
            observed_mask: optional mask of valid values, same shape as x
        Returns:
            scaled_x: x / scale
            scale: mean absolute value along dim
        """
        if observed_mask is not None:
            masked_x = x * observed_mask
            scale = masked_x.abs().sum(dim=self.dim, keepdim=self.keepdim)
            count = observed_mask.sum(dim=self.dim, keepdim=self.keepdim).clamp(min=1.0)
            scale = scale / count
        else:
            scale = x.abs().mean(dim=self.dim, keepdim=self.keepdim)

        # avoid dividing by zero
        scale = scale.clamp_min(self.eps)
        scaled_x = x / scale
        return scaled_x, scale

class ScoreDiffBase(pl.LightningModule):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        normalization="none",
        lr: float = 1e-3,
        dropout_rate: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dropout_rate = dropout_rate # reg method 1: dropout
        self.timesteps = timesteps
        
        self.betas = diffusion_scheduler(timesteps)
        self.sqrt_one_minus_beta = torch.sqrt(1.0 - self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.logs = {}
        self.normalization = normalization

        if normalization == "mean":
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        self.context_length = context_length
        self.prediction_length = prediction_length
        
        self.losses_running_mean = torch.ones(timesteps, requires_grad=False)
        self.lr = lr
        self.best_crps = np.inf


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(1e12)
        )
        return [optimizer], {"scheduler": scheduler, "monitor": "train_loss"}

    def log(self, name, value, **kwargs):
        super().log(name, value, **kwargs)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if name not in self.logs:
            self.logs[name] = [value]
        else:
            self.logs[name].append(value)

    def get_logs(self):
        logs = self.logs
        logs["epochs"] = list(range(self.current_epoch))
        return pd.DataFrame.from_dict(logs)

    def q_sample(self, x_start, t, noise=None):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        loss_type="l2",
        reduction="mean",
    ):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_score = self.backbone(x_noisy, t)  

        if loss_type == "l1":
            loss = F.l1_loss(predicted_score, noise, reduction=reduction)
        elif loss_type == "l2":
            loss = F.mse_loss(predicted_score, noise, reduction=reduction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(
                predicted_score, noise, reduction=reduction
            )
        else:
            raise NotImplementedError()

        return loss, x_noisy, predicted_score



    @torch.no_grad()
    def p_sample(self, x, t, t_index, y, h_fn, R_inv, base_strength=1.0,cheap=True):
        #given learnt score, predict unconditional model mean and then guide it using score of p(y_t|x_t) from tweedie approximation 
        betas_t = extract(self.betas, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        x, _ = self.scaler(x)
        
        predicted_score = self.backbone(x, t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_score / sqrt_one_minus_alphas_cumprod_t
        )
        
        if cheap:
            with torch.enable_grad():
                grad_logp_y = self.observation_grad_cheap(x, t, y, h_fn, R_inv)
            grad_logp_y = grad_logp_y.clamp(-1.0, 1.0)
            
                
            
        else:
            with torch.enable_grad():
                grad_logp_y = self.observation_grad_expensive(x,t,y,h_fn,R_inv)
            grad_logp_y = grad_logp_y.clamp(-1.0, 1.0)
        
            
        guide_strength = base_strength
        
        guided_mean = model_mean + guide_strength * betas_t * grad_logp_y


        if t_index == 0:
            sample = guided_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            sample = guided_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
        return sample

    def observation_grad_cheap(self, x_t, t, y, h_fn, R_inv):
        # no autograd through backbone required (only need grad through h)
        x_t = x_t.requires_grad_(True)
        eps = self.backbone(x_t, t)
        sqrt_bar_alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_bar = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        x0 = (x_t - sqrt_one_minus_bar * eps) / sqrt_bar_alpha
        y_pred = h_fn(x0)
        resid = y - y_pred
        r = R_inv(resid)

        # compute J_h^T r per-batch (still per-batch loop, but cheaper since no second vjp)
        Jt_r = []
        for i in range(x0.shape[0]):
            scalar = (y_pred[i].reshape(-1) * r[i].reshape(-1)).sum()
            gi = torch.autograd.grad(scalar, x0, retain_graph=True, create_graph=False)[0][i]
            Jt_r.append(gi)
        Jt_r = torch.stack(Jt_r, dim=0)
        return Jt_r

    def observation_grad_expensive(self, x_t, t, y, h_fn, R_inv):
        x_t = x_t.requires_grad_(True)
        eps = self.backbone(x_t, t)
        sqrt_bar_alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_bar = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        x0 = (x_t - sqrt_one_minus_bar * eps) / sqrt_bar_alpha
        y_pred = h_fn(x0)
        resid = y - y_pred
        r = R_inv(resid)                          # [B, ydim]

        # 1) compute w = J_h^T r  (vjp)
        # Do per-batch vjp for h: scalarize inner-product and grad wrt x0
        w = []
        for i in range(x0.shape[0]):
            scalar = (h_fn(x0[i]).flatten() * r[i].flatten()).sum()
            wi = torch.autograd.grad(scalar, x0, retain_graph=True, create_graph=True)[0][i]
            w.append(wi)
        w = torch.stack(w, dim=0)                 # [B, state_dim]

        # 2) compute v = (∂ε/∂x_t)^T w  (vjp through eps wrt x_t)
        # scalarize: (eps * w).sum() then grad wrt x_t
        v = []
        for i in range(x_t.shape[0]):
            scalar2 = (eps[i].flatten() * w[i].flatten()).sum()
            vi = torch.autograd.grad(scalar2, x_t, retain_graph=True, create_graph=False)[0][i]
            v.append(vi)
        v = torch.stack(v, dim=0)                 # [B, state_dim]

        # 3) combine
        obs_grad_wrt_xt = (w - sqrt_one_minus_bar * v) / sqrt_bar_alpha
        return obs_grad_wrt_xt


    @torch.no_grad()
    def sample(self, noise, y, h_fn, R_inv):
        device = next(self.backbone.parameters()).device
        batch_size, length, ch = noise.shape
        seq = noise
        seqs = [seq.cpu()]

        for i in reversed(range(0, self.timesteps)):
            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i,
                y,
                h_fn,
                R_inv
            )
            seqs.append(seq.cpu().numpy())

        return np.stack(seqs, axis=0)

    def fast_denoise(self, xt, t):
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, xt.shape)
        score = self.backbone(xt, t)
        return (xt - sqrt_one_minus_alphas_cumprod_t * score) / sqrt_alphas_cumprod_t

    def fast_sample(self,y,num_steps=None):
        device = next(self.backbone.parameters()).device
        batch_size, seq_len, ch = y.shape

        if num_steps is None:
            num_steps = self.timesteps
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            # Linear DDIM sampling schedule
            timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        # Initialize with standard normal noise
        x = torch.randn(batch_size, seq_len, ch, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            # fast deterministic denoising
            x = self.fast_denoise(x, t_tensor)
        
        return x

    def forward(self, x, mask):
        raise NotImplementedError()

    def training_step(self, data, idx):
        #learn the unconditional score of x to do sampling of
        device = next(self.backbone.parameters()).device

        # y: observations
        y = torch.as_tensor(data["past_observation"], dtype=torch.float32, device=device)
        y_future = torch.as_tensor(data["future_observation"], dtype=torch.float32, device=device)
        y_full = torch.cat([y, y_future], dim=1)

        # Initialize latent "states" as scaled versions of observations or learned embedding
        x_start, scale = self.scaler(y_full)

        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=device).long()

        loss_uncond, xt, predicted_noise = self.p_losses(x_start, t, loss_type="l2")
        self.log("elbo_loss", loss_uncond, prog_bar=True, on_step=True, on_epoch=True)
        return {
            "loss": loss_uncond,
            "elbo_loss": loss_uncond,
        }

    def training_epoch_end(self, outputs):
        epoch_loss = sum(x["loss"] for x in outputs) / len(outputs)
        elbo_loss = sum(x["elbo_loss"] for x in outputs) / len(outputs)
        self.log("train_loss", epoch_loss)
        self.log("train_elbo_loss", elbo_loss)

    def validation_step(self, data, idx):
        device = next(self.backbone.parameters()).device
        x = data
        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss, xt, noise = self.p_losses(x, t, loss_type="l2")
        return {
            "loss": elbo_loss,
            "elbo_loss": elbo_loss,
        }

    def validation_epoch_end(self, outputs):
        epoch_loss = sum(x["loss"] for x in outputs) / len(outputs)
        elbo_loss = sum(x["elbo_loss"] for x in outputs) / len(outputs)
        self.log("valid_loss", epoch_loss)
        self.log("valid_elbo_loss", elbo_loss)
