from models.Neurons import LIF
from models.Populations import *
from models.Synapses import Synapse
from models.Connections import Connection
from simulation.Simulate import Simulate
from simulation.Monitors import raster_plot


def current_generator(duration, dt, val):
    current_list = []
    for i in np.arange(0, duration, dt):
        if i < 5:
            current_list.append(0)
        elif i < duration - 5:
            current_list.append(val)
        else:
            current_list.append(0)
    return current_list


duration = 10000
dt = 1
neuron_params = {
    "tau": 10,
    "r": 2,
    "threshold": -50,
    "current": current_generator(duration, dt, 0)
}

stdp_params = {
    "a_plus": lambda x: 10,
    "a_minus": lambda x: -10,
    "tau_plus": 5,
    "tau_minus": 2
}

input_pop = InputPopulation(10, interval=10)
output_pop = Population(2, LIF, **neuron_params)
inputs = [
    [1, 3, 0, 2, 0, 4, 1, 0, 0, 0],
    [5, 0, 0, 0, 7, 3, 0, 2, 0, 6]
]
input_pop.set_input(inputs)
conn = Connection(input_pop, output_pop).apply(
    "full", mu=5, sigma=0.1, **stdp_params)
sim = Simulate(populations=[input_pop, output_pop],
               connections=[conn], time_step=dt)
sim.run(duration, learning_rule="stdp")
for syn in conn.synapses:
    print(syn.pre.name, "--{}-->".format(syn.w), syn.post.name)

one = output_pop.spikes_per_neuron[output_pop.spikes_per_neuron[:, 0] == 0, 1]
two = output_pop.spikes_per_neuron[output_pop.spikes_per_neuron[:, 0] == 1, 1]
print(one, two)

raster_plot(output_pop.spikes_per_neuron)
