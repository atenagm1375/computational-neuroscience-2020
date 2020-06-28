import matplotlib.pyplot as plt
from models.Neurons import *
from models.Synapses import Synapse

import numpy as np


def get_current(duration, dt, delta):
    src = []
    dest = []
    if delta >= 0:
        for i in np.arange(0, duration, dt):
            src.append(5)
            if i >= delta:
                dest.append(5)
            else:
                dest.append(0)
    else:
        for i in np.arange(0, duration, dt):
            dest.append(5)
            if i >= -1 * delta:
                src.append(5)
            else:
                src.append(0)
    return src, dest


dt = 1
duration = 20
delta_w_delta_t = []
for i in np.arange(-1 * duration, duration, 0.5):
    curr_src, curr_dest = get_current(duration, dt, i)
    src = LIF(tau=8, current=curr_src)
    dest = LIF(tau=8, current=curr_dest)
    stdp_params = {
        "a_plus": lambda x: 10,
        "a_minus": lambda x: -10,
        "tau_plus": 5,
        "tau_minus": 5
    }
    syn = Synapse(src, dest, 5, **stdp_params)

    src.input = np.zeros(int(duration // dt))
    dest.input = np.zeros(int(duration // dt))

    for t in np.arange(0, duration, dt):
        src.compute_potential(t, dt)
        src.apply_pre_synaptic(t, dt)
        src.compute_spike(t, dt)
        src.reset(t, dt, 0.5)

        dest.compute_potential(t, dt)
        dest.apply_pre_synaptic(t, dt)
        dest.compute_spike(t, dt)
        dest.reset(t, dt, 0.5)

        if len(src.spike_times) > 0 and len(dest.spike_times) > 0:
            delta_t, delta_w = syn.update("stdp", t, dt)
            delta_w_delta_t.append([delta_t, delta_w])
            break


# print(delta_w_delta_t)

delta_w_delta_t = np.array(delta_w_delta_t)
plt.plot(delta_w_delta_t[:, 0], delta_w_delta_t[:, 1], 'o')
plt.show()
