from models.Neurons import LIF
from models.Populations import Population, InputPopulation2
from models.Connections import Connection
from simulation.Simulate import Network
from simulation.Monitors import raster_plot

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
    reward = [0 for i in range(len(dest_neurons))]
    activities = [0 for i in range(len(dest_neurons))]
    for i, neurons in enumerate(dest_neurons):
        activities[i] += np.sum([x.spike_times[-1] == t for x in neurons if len(x.spike_times) > 0])
    most_active = np.argmax(activities)
    if isinstance(most_active, np.int64):
        reward[most_active] = 1 if src_seq[t] == most_active else -1
    elif src_seq[t] in most_active:
        for i in most_active:
            reward[i] = 1/len(most_active) if i == src_seq else -1/len(most_active)
    else:
        reward = [-1 for i in most_active]

    dopamine_change = np.sum([0.5, -0.5] * np.array(reward))
    return dopamine_change


duration = 2000
dt = 1

neuron_params = {
    "tau": 10,
    "r": 2,
    "threshold": -60,
    "current": current_generator(duration, dt, 1, 5, 100)
}

rstdp_params = {
    "a_plus": lambda x: 0.02 * (20 - x),
    "a_minus": lambda x: 0.02 * (-20 - x),
    "tau_plus": 8,
    "tau_minus": 8,
    "c": 5,
    "tau_c": 1000,
    "tau_d": 20
}

pop = Population(1000, LIF, exc_ratio=0.8, **neuron_params)

ins, outs = pop.choose_input_output((1, 50), (2, 20))

input_pattern = [
    np.random.randint(0, 4, 50),
    np.random.randint(0, 4, 50)
]

pop.encode_input(input_pattern, np.array(ins), duration, 15)

conn = Connection(pop, pop).apply("fixed_pre", p=0.1, mu=1.5, sigma=0.25, **rstdp_params)

weight_matrix = np.zeros((pop.size, pop.size))
for syn in conn.synapses:
    weight_matrix[pop.neurons.index(syn.pre), pop.neurons.index(syn.post)] = syn.w
# print(weight_matrix)

net = Network(populations=[pop], connections=[conn], time_step=dt)
net.set_input_output_neurons(ins, outs)
net.set_dopamine(1.5, func_da)
net.run(duration, learning_rule="rstdp")

for syn in conn.synapses:
    weight_matrix[pop.neurons.index(syn.pre), pop.neurons.index(syn.post)] = syn.w

# print(weight_matrix)
# raster_plot(src.spikes_per_neuron)
# raster_plot(dest.spikes_per_neuron)
sns.heatmap(weight_matrix)
plt.show()
