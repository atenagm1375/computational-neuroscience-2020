from models.Neurons import LIF
from models.Populations import Population, InputPopulation2
from models.Connections import Connection
from simulation.Simulate import Network
from simulation.Monitors import plot_weight_matrix, plot_weight_change

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


def main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         n_params, s_params, regularize=False, plot_change=False):
    src = InputPopulation2(src_size, LIF, **n_params)
    n_params["current"] = list(np.zeros(duration // dt))
    n_params["regularize"] = regularize
    dest = Population(dest_size, LIF, **n_params)

    src.encode(input_pattern, duration, interval)

    conn = Connection(src, dest).apply(**s_params)

    net = Network(populations=[src, dest], connections=[conn], time_step=dt)
    net.set_dopamine(initial_dopamine, func_da)
    net.run(duration, learning_rule="rstdp")

    print(conn.weight_in_time[-1])
    plot_weight_matrix(conn.weight_in_time[-1])
    if plot_change:
        plot_weight_change(conn.weight_in_time)


def trial1():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t and src_seq[t] >= 0:
                reward[i] = -1 if src_seq[t] != i else 1
        dopamine_change = np.sum(0.8 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.1
        return dopamine_change

    duration = 15000
    dt = 1

    neuron_params = {
        "tau": 10,
        "r": 2,
        "threshold": -60,
        "current": current_generator(duration, dt, 0.5, 5, 100)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 1.6,
        "sigma": 0.25,
        "a_plus": lambda x: 0.5 if x < 20 else 0.001,
        "a_minus": lambda x: -0.5 if x > -20 else -0.001,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.5,
        "tau_c": 1000,
        "tau_d": 20
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 20
    initial_dopamine = 0.8
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params)


def trial2():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t and src_seq[t] >= 0:
                reward[i] = -1 if src_seq[t] != i else 1
        dopamine_change = np.sum(0.8 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.1
        return dopamine_change

    duration = 15000
    dt = 1

    neuron_params = {
        "tau": 10,
        "r": 2,
        "threshold": -60,
        "current": current_generator(duration, dt, 0.5, 5, 100)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 1.6,
        "sigma": 0.25,
        "a_plus": lambda x: 0.5 if x < 20 else 0.001,
        "a_minus": lambda x: -0.5 if x > -20 else -0.001,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.5,
        "tau_c": 1000,
        "tau_d": 100
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 20
    initial_dopamine = 0.8
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params)


def trial3():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t and src_seq[t] >= 0:
                reward[i] = -1 if src_seq[t] != i else 1
        dopamine_change = np.sum(0.8 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.1
        return dopamine_change

    duration = 15000
    dt = 1

    neuron_params = {
        "tau": 10,
        "r": 2,
        "threshold": -60,
        "current": current_generator(duration, dt, 0.5, 5, 100)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 1.6,
        "sigma": 0.25,
        "a_plus": lambda x: 0.5 if x < 20 else 0.001,
        "a_minus": lambda x: -0.5 if x > -20 else -0.001,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.5,
        "tau_c": 500,
        "tau_d": 20
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 20
    initial_dopamine = 0.8
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params)


def trial4():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t and src_seq[t] >= 0:
                reward[i] = -1 if src_seq[t] != i else 1
        dopamine_change = np.sum(0.5 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.01
        return dopamine_change

    duration = 15000
    dt = 1

    neuron_params = {
        "tau": 10,
        "r": 2,
        "threshold": -60,
        "current": current_generator(duration, dt, 0.5, 5, 100)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 1.6,
        "sigma": 0.25,
        "a_plus": lambda x: 0.5 if x < 20 else 0.001,
        "a_minus": lambda x: -0.5 if x > -20 else -0.001,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.5,
        "tau_c": 1000,
        "tau_d": 20
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 20
    initial_dopamine = 0.2
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params)
