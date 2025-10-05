
import numpy as np

def trainingData(function,samples=10, context_length=100, dt=1,q=1):
    true_states = []
    noisy_states = []
    for sample in range(samples):
        true_state, noisy_state = function(context_length=context_length, dt=dt,q=q)

        true_states.append(true_state)
        noisy_states.append(noisy_state)
    return np.array(noisy_states), np.array(true_states)

def sinusoidalWaves(context_length=100,dt=1,q=1):
    xs= []
    ys= []
    amplitude=np.random.uniform(1,5)
    frequency=np.random.uniform(1,4)
    phase=np.random.uniform(0,2*np.pi)
    for step in range(context_length*(int(dt**-1))):
        w = np.random.normal(0, q)
        x = np.sin(frequency*step*dt + phase)*amplitude
        xs.append(x)
        y = x + w
        ys.append(y)
    return xs, ys