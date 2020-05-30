from models.Neurons import LIF
from models.Populations import Population, InputPopulation2
from models.Connections import Connection
from simulation.Simulate import Simulate

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


duration = 10000
dt = 1

neuron_params = {
    "tau": 10,
    "r": 2,
    "threshold": -55,
    "current": current_generator(duration, dt, 2, 5, 10)
}

rstdp_params = {
    "a_plus": 2,
    "a_minus": -2,
    "tau_plus": 6,
    "tau_minus": 6,
    "c": 2,
    "d": 2,
    "tau_c": 100,
    "tau_d": 10,
    "da": None
}

src = InputPopulation2(10, LIF, **neuron_params)
dest = Population(2, LIF, **neuron_params)

input_pattern = [
    [1, 2, 2, 1, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 3, 1, 1, 1]
]

src.encode(input_pattern, duration, 5)

conn = Connection(src, dest).apply("full", mu=1.75, sigma=0.15, **rstdp_params)

weight_matrix = np.zeros((src.size, dest.size))
for syn in conn.synapses:
    weight_matrix[src.neurons.index(
        syn.pre), dest.neurons.index(syn.post)] = syn.w
print(weight_matrix)

sim = Simulate(populations=[src, dest],
               connections=[conn], time_step=dt)
sim.run(duration, learning_rule="rstdp")

for syn in conn.synapses:
    weight_matrix[src.neurons.index(
        syn.pre), dest.neurons.index(syn.post)] = syn.w

print(weight_matrix)
