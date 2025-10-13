# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from dataGeneration import SinusoidalWaves
import matplotlib.pyplot as plt
import random

def get_custom_dataset(dataset_name, samples=10, context_length=80,prediction_length=20, dt=1,q=1,r=1,observation_dim=1,plot=False):
    generatingClasses = {
        "sinusoidal": SinusoidalWaves,
    }
    generator = generatingClasses[dataset_name.split(":")[1]](context_length+prediction_length,dt,q,r,observation_dim)

    states = []
    observations = []
    for sample in range(samples):
        state, obs = generator.generate()
        states.append(state)
        observations.append(obs)
        
    state_array = np.array(states)
    observation_array = np.array(observations)
    h_fn = generator.h_fn
    R_inv = generator.R_inv

    if plot:
        fig = plt.figure(figsize=(10, 6))
        plt.title('Sinusoidal Wave with Noisy Observations')
        plt.axis('off')
        axis = fig.add_subplot(111)
        true_states = random.sample(states, k=10)
        noisy_states = random.sample(observations, k=10)

        dataRange = np.arange(0, context_length+prediction_length, dt)
        for true_state, noisy_state in zip(true_states, noisy_states):
            axis.plot(dataRange,true_state)
        axis.axvline(x=context_length, linestyle=':', color='r', label=f'End of context')
        axis.legend()

        plt.show()



    custom_data = [
        {
            "state": np.array(state)[..., np.newaxis],         # shape (seq_len, 1)
            "observation": np.array(obs)[..., np.newaxis],    # shape (seq_len, 1)
        }
        for obs, state in zip(observation_array, state_array)
    ]
    return np.array(custom_data), generator
