from models.Neurons import LIF
from models.Populations import *
from models.Synapses import Synapse
from models.Connections import Connection
from simulation.Simulate import Network
from simulation.Monitors import raster_plot


def current_generator(duration, dt, val, begin=5):
    current_list = []
    for i in np.arange(0, duration, dt):
        if i < begin:
            current_list.append(0)
        elif i < duration / 2:
            current_list.append(val)
        else:
            current_list.append(0)
    return current_list


duration = 10000
dt = 1
neuron_params = {
    "tau": 10,
    "r": 4,
    "threshold": -60,
    "current": current_generator(duration, dt, 0.5, 10)
}

input_params = {
    "tau": 1,
    "r": 5,
    "u_rest": 0,
    "threshold": 5,
    "current": current_generator(duration, dt, 0.5)
}

stdp_params = {
    "a_plus": lambda x: 1 * (40 - x),
    "a_minus": lambda x: 1 * (-40 - x),
    "tau_plus": 1,
    "tau_minus": 1
}

input_pop = InputPopulation(10, LIF, **input_params)
output_pop = Population(1, LIF, trace_alpha=0.1, **neuron_params)
neuron_params["current"] = current_generator(duration, dt, 0.5, 8)
output_pop.add(1, LIF, **neuron_params)
inputs = [
    [1, 2, 2, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 1, 2, 1, 1]
]
input_pop.encode(inputs, duration, 6)
conn = Connection(input_pop, output_pop).apply(
    "full", mu=1.75, sigma=0.15, **stdp_params)

inh_params = {
    "tau": 10,
    "r": 5,
    "threshold": -60,
    "is_inh": True,
    "current": current_generator(duration, dt, 0.75, 9)
}
output_pop.add(1, LIF, **inh_params)
conn.add(list(range(input_pop.size)),
         [-1], mu=1.25, sigma=0.1, delay=0, **stdp_params)
conn.add([-1], [0, 1], mu=0.8, sigma=0.1, delay=0, **stdp_params)

weight_matrix = np.zeros((input_pop.size, output_pop.size))
for syn in conn.synapses:
    weight_matrix[input_pop.neurons.index(
        syn.pre), output_pop.neurons.index(syn.post)] = syn.w
print(weight_matrix)

sim = Network(populations=[input_pop, output_pop],
               connections=[conn], time_step=dt)
sim.run(duration, learning_rule="stdp")

for syn in conn.synapses:
    weight_matrix[input_pop.neurons.index(
        syn.pre), output_pop.neurons.index(syn.post)] = syn.w
# print(syn.pre.name, "--{}-->".format(syn.w), syn.post.name)

print(weight_matrix)

# for neuron in input_pop.neurons:
#     print(neuron.current_list)
#
# for i in range(0, 10):
#     print(
#         input_pop.spikes_per_neuron[input_pop.spikes_per_neuron[:, 0] == i, 1])
#
# if output_pop.spikes_per_neuron.size > 0:
#     one = output_pop.spikes_per_neuron[output_pop.spikes_per_neuron[:, 0] == 0, 1]
#     two = output_pop.spikes_per_neuron[output_pop.spikes_per_neuron[:, 0] == 1, 1]
#     print(one, two)

# raster_plot(output_pop.spikes_per_neuron)
