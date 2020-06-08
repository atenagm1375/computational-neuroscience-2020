from copy import deepcopy

from models.Neurons import LIF
from models.Populations import Population, InputPopulation2
from models.Connections import Connection
from simulation.Simulate import Network
from simulation.Monitors import activity_plot, raster_plot

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


def main(duration, dt, population_size, n, input_size, output_size, interval, initial_dopamine,
         func_da, neuron_params, rstdp_params):

    inp = InputPopulation2(input_size, LIF, **neuron_params)
    input_pattern = []
    for i in range(n):
        input_pattern.append(np.random.randint(0, 3, input_size))
    inp.encode(input_pattern, duration, interval, dt)

    outs = []
    neuron_params["regularize"] = True
    for i in range(n):
        outs.append(Population(output_size, LIF, trace_alpha=0, **neuron_params))
    pop = Population(population_size, LIF, input_part=inp, output_part=outs, exc_ratio=0.8,
                     trace_alpha=0.5, **neuron_params)

    # ins, outs = pop.choose_input_output((1, input_size), (n, output_size))

    # pop.encode_input(input_pattern, np.array(ins), duration, interval)

    conn = Connection(pop, pop).apply(**rstdp_params)

    net = Network(populations=[pop], connections=[conn], time_step=dt)
    # net.set_input_output_neurons(ins, outs)
    net.set_dopamine(initial_dopamine, func_da)
    net.run(duration, learning_rule="rstdp")
    # inp.compute_spike_history()
    # raster_plot(np.array(inp.spikes_per_neuron))
    activity_plot([out.activity for out in outs])


def trial1():
    # noinspection PyTypeChecker
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        most_active = np.argmax([out.activity[-1][1] for out in pop.output_part])
        reward[most_active] = 1 if pop.input_part.input_seq[t] == most_active else -1 / (len(pop.output_part) - 1)
        for i in range(len(reward)):
            if i != most_active:
                reward[i] = -1 / (len(pop.output_part) - 1)

        # print(reward)
        dopamine_change = np.sum(0.8 * np.array(reward))
        return dopamine_change

    duration = 15000
    dt = 1

    neuron_params = {
        "tau": 8,
        "r": 2,
        "threshold": -62,
        "current": current_generator(duration, dt, 2, 5, 100)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.5,
        "mu": 1.75,
        "sigma": 0.25,
        "a_plus": lambda x: 0.5 if x < 20 else 0.001,
        "a_minus": lambda x: -0.5 if x > -20 else -0.001,
        "tau_plus": 6,
        "tau_minus": 6,
        "c": 0.8,
        "tau_c": 800,
        "tau_d": 18
    }

    main(duration, dt, 100, 2, 5, 2, 20, 0.8, func_da, neuron_params, rstdp_params)
