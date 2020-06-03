from models.Neurons import LIF
from models.Populations import Population, InputPopulation2
from models.Connections import Connection
from simulation.Simulate import Network
from simulation.Monitors import raster_plot, plot_weight_change

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def current_generator(duration, dt, val, begin=5, ratio=8):
    current_list = []
    for i in np.arange(0, duration, dt):
        if i < begin:
            current_list.append(0)
        elif i < duration / ratio:
            current_list.append(val)
        else:
            current_list.append(0)
    return current_list


def func_da(src_seq, dest_neurons, t):
    reward = [0 for _ in range(len(dest_neurons))]
    for i in range(len(dest_neurons)):
        if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t and src_seq[t] >= 0:
            reward[i] = -1 if src_seq[t] != i else 1
    dopamine_change = np.sum(0.8 * np.array(reward))
    if np.prod(np.array(reward)) < 0:
        dopamine_change -= 0.1
    return dopamine_change


duration = 12000
dt = 1

neuron_params = {
    "tau": 10,
    "r": 2,
    "threshold": -60,
    "current": current_generator(duration, dt, 1, 5, 100)
}

rstdp_params = {
    "a_plus": lambda x: 0.5 if x < 20 else 0.001,
    "a_minus": lambda x: -0.5 if x > -20 else -0.001,
    "tau_plus": 8,
    "tau_minus": 8,
    "c": 0.5,
    "tau_c": 1000,
    "tau_d": 20
}

src = InputPopulation2(10, LIF, **neuron_params)
neuron_params["current"] = current_generator(duration, dt, 3, 5, 1000)
dest = Population(2, LIF, **neuron_params)

input_pattern = [
    [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
]

src.encode(input_pattern, duration, 20)

conn = Connection(src, dest).apply("full", mu=1.5, sigma=0.25, **rstdp_params)

weight_matrix = np.zeros((src.size, dest.size))
for syn in conn.synapses:
    weight_matrix[src.neurons.index(syn.pre), dest.neurons.index(syn.post)] = syn.w
print(weight_matrix)

net = Network(populations=[src, dest], connections=[conn], time_step=dt)
net.set_dopamine(0.8, func_da)
net.run(duration, learning_rule="rstdp")

for syn in conn.synapses:
    weight_matrix[src.neurons.index(syn.pre), dest.neurons.index(syn.post)] = syn.w

print(weight_matrix)
# raster_plot(src.spikes_per_neuron)
# raster_plot(dest.spikes_per_neuron)
sns.heatmap(weight_matrix)
plt.show()
plot_weight_change(conn.weight_in_time)
