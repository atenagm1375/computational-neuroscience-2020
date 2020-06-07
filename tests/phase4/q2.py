from copy import deepcopy

from models.Neurons import LIF
from models.Populations import Population
from models.Connections import Connection
from simulation.Simulate import Network
from simulation.Monitors import activity_plot

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
    pop = Population(population_size, LIF, exc_ratio=0.8, trace_alpha=0, **neuron_params)

    ins, outs = pop.choose_input_output((1, input_size), (n, output_size))

    input_pattern = []
    for i in range(n):
        input_pattern.append(np.random.randint(0, 3, input_size))

    pop.encode_input(input_pattern, np.array(ins), duration, interval)

    conn = Connection(pop, pop).apply(**rstdp_params)

    net = Network(populations=[pop], connections=[conn], time_step=dt)
    net.set_input_output_neurons(ins, outs)
    net.set_dopamine(initial_dopamine, func_da)
    net.run(duration, learning_rule="rstdp")
    activity_plot(pop.output_activity, outs, pop.input_seq)


def trial1():
    # noinspection PyTypeChecker
    def func_da(src_seq, dest_neurons, activities, t):
        reward = [0 for _ in range(len(dest_neurons))]
        activities = np.array(activities)
        activities = activities[:, t, 1]
        most_active = np.argmax(activities)
        reward[most_active] = 1 if src_seq[t] == most_active else -1

        # print(reward)
        dopamine_change = np.sum(0.8 * np.array(reward))
        return dopamine_change

    duration = 12000
    dt = 1

    neuron_params = {
        "tau": 10,
        "r": 2,
        "threshold": -62,
        "current": current_generator(duration, dt, 2, 5, 100)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.5,
        "mu": 1.6,
        "sigma": 0.25,
        "a_plus": lambda x: 0.2 if x < 20 else 0.001,
        "a_minus": lambda x: -0.2 if x > -20 else -0.001,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.5,
        "tau_c": 1000,
        "tau_d": 20
    }

    main(duration, dt, 100, 2, 5, 2, 50, 0.5, func_da, neuron_params, rstdp_params)
