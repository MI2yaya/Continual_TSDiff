# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import argparse
from pathlib import Path

import yaml
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.dataset import get_custom_dataset
from uncond_ts_diff.model.sfcallback import SFPredictiveScoreCallback, StateMSECallback
from uncond_ts_diff.model.diffusion.sfdiff import SFDiff
from uncond_ts_diff.utils import (
    add_config_to_argparser,
    filter_metrics,
    train_test_val_splitter,
    time_splitter,
    StateObsDataset
)
import uncond_ts_diff.utils
from torch.utils.data import TensorDataset, DataLoader
import importlib


importlib.reload(uncond_ts_diff.utils)



def create_model(config,context_length,prediction_length):
    model = SFDiff(
        **getattr(diffusion_configs, config["diffusion_config"]),
        state_dim = config["state_dim"],
        observation_dim=config["observation_dim"],
        normalization=config["normalization"],
        context_length=context_length,
        prediction_length=prediction_length,
        lr=config["lr"],
        init_skip=config["init_skip"]
    )
    model.to(config["device"])
    return model


def main(config, log_dir):
    dataset_name = config["dataset"]
    scaling = int(config['dt']**-1)

    context_length = config["context_length"] * scaling
    prediction_length = config["prediction_length"] * scaling
    total_length = context_length + prediction_length

    model = create_model(config,context_length,prediction_length)
    dataset = get_custom_dataset(dataset_name,samples=config['data_samples'],context_length=config["context_length"],prediction_length=config["prediction_length"],dt=config['dt'],q=config['q'])

    # Split dataset
    #print(dataset)
    time_data = time_splitter(dataset, context_length, prediction_length)
    split_data = train_test_val_splitter(time_data, config['data_samples'], .6, .4, 0.0)
    # Prepare training data
    train_dataset = StateObsDataset(split_data["train"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Callbacks
    callbacks = [
        StateMSECallback(
            context_length=context_length,
            prediction_length=prediction_length,
            model=model,
            test_dataset = split_data["test"],
            test_batch_size=config['num_samples'],
            eval_every=config['eval_every']
        )
    ]

    checkpoint_callback = ModelCheckpoint(
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            filename=f"{dataset_name.replace(':','_')}-{{epoch:03d}}-{{train_loss:.3f}}",
            save_last=True,
            save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)

    # Trainer setup
    if config["device"].startswith("cuda") and torch.cuda.is_available():
        devices = [int(config["device"].split(":")[-1])]
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
        gradient_clip_val=config.get("gradient_clip_val", None),
    )


    logger.info(f"Logging to {trainer.logger.log_dir}")
    trainer.fit(model, train_loader)
    logger.info("Training completed and best checkpoint saved.")
    best_ckpt_path = Path(trainer.logger.log_dir) / "best_checkpoint.ckpt"

    if not best_ckpt_path.exists():
        torch.save(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
            best_ckpt_path,
        )



if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./", help="Path to results dir"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    main(config=config, log_dir=args.out_dir)
