# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import tarfile
from pathlib import Path
from urllib import request
import numpy as np

from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository.datasets import get_dataset, get_download_path
from gluonts.dataset.common import ListDataset
from dataGeneration import trainingData, sinusoidalWaves
import matplotlib.pyplot as plt
import random

default_dataset_path: Path = get_download_path() / "datasets"
wiki2k_download_link: str = "https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz"  # noqa: E501

def get_custom_dataset(dataset_name, samples=10, context_length=80,prediction_length=20, dt=1,q=1,plot=False):
    functions = {
        "sinusoidal": sinusoidalWaves,
    }
    function = functions[dataset_name.split(":")[1]]

    obs_array, state_array = trainingData(
        function=function, samples=samples, context_length=context_length+prediction_length, dt=dt, q=q
    )
    if plot:
        fig = plt.figure(figsize=(10, 6))
        plt.title('Sinusoidal Wave with Noisy Observations')
        plt.axis('off')
        axis = fig.add_subplot(111)
        true_states = random.sample(list(state_array), k=10)
        noisy_states = random.sample(list(obs_array), k=10)

        dataRange = np.arange(0, context_length+prediction_length, dt)
        for true_state, noisy_state in zip(true_states, noisy_states):
            axis.plot(dataRange,true_state)
        axis.axvline(x=context_length, linestyle=':', color='r', label=f'End of context')
        axis.legend()

        plt.show()

    custom_data = [
        {
            "start": "2020-01-01",
            "target": np.array(state),          
            "observed_values": np.array(obs),     
            "feat_static_cat": None,
            "feat_static_real": None,
        }
        for obs, state in zip(obs_array, state_array)
    ]
    return ListDataset(data_iter=custom_data, freq="H")

def get_gts_dataset(dataset_name):
    if dataset_name == "wiki2000_nips":
        wiki_dataset_path = default_dataset_path / dataset_name
        Path(default_dataset_path).mkdir(parents=True, exist_ok=True)
        if not wiki_dataset_path.exists():
            tar_file_path = wiki_dataset_path.parent / f"{dataset_name}.tar.gz"
            request.urlretrieve(
                wiki2k_download_link,
                tar_file_path,
            )

            with tarfile.open(tar_file_path) as tar:
                tar.extractall(path=wiki_dataset_path.parent)

            os.remove(tar_file_path)
        return load_datasets(
            metadata=wiki_dataset_path / "metadata",
            train=wiki_dataset_path / "train",
            test=wiki_dataset_path / "test",
        )
    
    else:
        return get_dataset(dataset_name)
