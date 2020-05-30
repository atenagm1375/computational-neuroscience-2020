from models.Neurons import LIF
from models.Populations import Population, InputPopulation2
from models.Connections import Connection
from simulation.Simulate import Network

import numpy as np


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
    if len(src_seq) > 0:
        for i in range(len(dest_neurons)):
            if dest_neurons[i].spike_times[-1] == t and src_seq[0] == i:
                src_seq.pop(0)
                return 10
            if dest_neurons[i].spike_times[-1] == t and src_seq[0] != i:
                src_seq.pop(0)
                return -10
    return 0


duration = 10000
dt = 1

neuron_params = {
    "tau": 10,
    "r": 2,
    "threshold": -55,
    "current": current_generator(duration, dt, 2, 5, 100)
}

rstdp_params = {
    "a_plus": lambda x: (80 - x),
    "a_minus": lambda x: (-80 - x),
    "tau_plus": 6,
    "tau_minus": 6,
    "c": 10,
    "tau_c": 100,
    "tau_d": 10
}

src = InputPopulation2(10, LIF, **neuron_params)
dest = Population(2, LIF, **neuron_params)

input_pattern = [
    [1, 2, 2, 1, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 3, 1, 1, 1]
]

src.encode(input_pattern, duration, 10)

conn = Connection(src, dest).apply("full", mu=1.75, sigma=0.15, **rstdp_params)

weight_matrix = np.zeros((src.size, dest.size))
for syn in conn.synapses:
    weight_matrix[src.neurons.index(
        syn.pre), dest.neurons.index(syn.post)] = syn.w
print(weight_matrix)

net = Network(populations=[src, dest], connections=[conn], time_step=dt)
net.set_dopamine(10, func_da)
net.run(duration, learning_rule="rstdp")

for syn in conn.synapses:
    weight_matrix[src.neurons.index(
        syn.pre), dest.neurons.index(syn.post)] = syn.w

print(weight_matrix)
