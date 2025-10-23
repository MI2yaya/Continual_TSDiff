# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np

sns.set(
    style="white",
    font_scale=1.1,
    rc={"figure.dpi": 125, "lines.linewidth": 2.5, "axes.linewidth": 1.5},
)


def filter_metrics(metrics, select={"ND", "NRMSE", "mean_wQuantileLoss"}):
    return {m: metrics[m].item() for m in select}


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)


def plot_train_stats(df: pd.DataFrame, y_keys=None, skip_first_epoch=True):
    if skip_first_epoch:
        df = df.iloc[1:, :]
    if y_keys is None:
        y_keys = ["train_loss", "valid_loss"]

    fix, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for y_key in y_keys:
        sns.lineplot(
            ax=ax,
            data=df,
            x="epochs",
            y=y_key,
            label=y_key.replace("_", " ").capitalize(),
        )
    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    plt.show()

def get_next_file_num(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
):
    """Gets the next available file number in a directory.
    e.g., if `base_fname="results"` and `base_dir` has
    files ["results-0.yaml", "results-1.yaml"],
    this function returns 2.

    Parameters
    ----------
    base_fname
        Base name of the file.
    base_dir
        Base directory where files are located.

    Returns
    -------
        Next available file number
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and x.name.startswith(base_fname),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: x.name.startswith(base_fname),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    return max(run_nums) + 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def add_config_to_argparser(config: Dict, parser: ArgumentParser):
    for k, v in config.items():
        sanitized_key = re.sub(r"[^\w\-]", "", k).replace("-", "_")
        val_type = type(v)
        if val_type not in {int, float, str, bool}:
            print(f"WARNING: Skipping key {k}!")
            continue
        if val_type == bool:
            parser.add_argument(f"--{sanitized_key}", type=str2bool, default=v)
        else:
            parser.add_argument(f"--{sanitized_key}", type=val_type, default=v)
    return parser



def descale(data, scale, scaling_type):
    if scaling_type == "mean":
        return data * scale
    elif scaling_type == "min-max":
        loc, scale = scale
        return data * scale + loc
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")



def time_splitter(data, past_length, future_length):
    newDataset = []
    for arr in data:
        tempDict = {}
        for key, series in arr.items():
            if key=="state" or key=="observation":
                tempDict[f"past_{key}"] = series[:past_length]
                tempDict[f"future_{key}"] = series[past_length:past_length + future_length]
            else:
                tempDict[key] = series
        
        newDataset.append(tempDict)
    return np.array(newDataset, dtype=object)

def train_test_val_splitter(data,totalSamples,ptrain=.6,ptest=.2,pval=.2):
    assert abs(ptrain + ptest + pval - 1) < 1e-6, "Splits must sum to 1"
    trainIndex = int(totalSamples * ptrain)
    testIndex = trainIndex + int(totalSamples * ptest)
    return {
        "train": data[:trainIndex],
        "test": data[trainIndex:testIndex],
        "val": data[testIndex:] if pval > 0 else []
    }

from torch.utils.data import Dataset
class StateObsDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of dicts, each with "past_state" and "past_observation"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "past_state": torch.tensor(item["past_state"], dtype=torch.float32),
            "past_observation": torch.tensor(item["past_observation"], dtype=torch.float32),
            "future_state": torch.tensor(item["future_state"], dtype=torch.float32),
            "future_observation": torch.tensor(item["future_observation"], dtype=torch.float32),
        }
        
        
import imageio
import os
from natsort import natsorted
import shutil

def make_diffusion_gif(output_path="reverse_diffusion.gif", frame_dir="diffusion_frames", fps=20):
    frames = [imageio.imread(os.path.join(frame_dir, f)) 
              for f in natsorted(os.listdir(frame_dir)) if f.endswith(".png")]
    frames.reverse()
    pause_frames = int(3 * fps)
    frames += [frames[-1]] * pause_frames
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved GIF: {output_path}")
    shutil.rmtree(frame_dir)