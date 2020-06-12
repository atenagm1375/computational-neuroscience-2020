from models.Neurons import LIF
from models.Populations import Population, InputPopulation
from models.Connections import Connection
from simulation.Simulate import Network
from simulation.Monitors import plot_weight_matrix, plot_weight_change, raster_plot

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
         n_params, s_params, trace_alpha=0.1, regularize=None, plot_change=False):

    src = InputPopulation(src_size, LIF, **n_params)
    src.encode(input_pattern, duration, interval)

    n_params["current"] = list(np.zeros(duration // dt))
    n_params["regularize"] = regularize
    dest = Population(dest_size, LIF, trace_alpha=trace_alpha, **n_params)

    conn = Connection(src, dest).apply(**s_params)

    net = Network(populations=[src, dest], connections=[conn], time_step=dt)
    net.set_dopamine(initial_dopamine, func_da)
    net.run(duration, learning_rule="rstdp")

    # raster_plot(src.spikes_per_neuron)
    # raster_plot(dest.spikes_per_neuron)

    print(conn.weight_in_time[-1])
    plot_weight_matrix(conn.weight_in_time[-1])
    if plot_change:
        plot_weight_change(conn.weight_in_time)


def trial1():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        nonz = np.nonzero(src_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t:
                if ind >= 0:
                    reward[i] += -1 if int(src_seq[ind]) != i + 1 else 1
                    if i > 0:
                        print(reward)
        if ind >= 0:
            src_seq[ind-2:ind+1] = 0
        dopamine_change = np.sum(0.2 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.001
        return dopamine_change

    duration = 30000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 9,
        "sigma": 0.2,
        "w_min": -16,
        "w_max": 16,
        "a_plus": lambda x: 0.005,
        "a_minus": lambda x: -0.01,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.05,
        "tau_c": 200,
        "tau_d": 50
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 500
    initial_dopamine = 0.05
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params, plot_change=True)


def trial2():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        nonz = np.nonzero(src_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t:
                if ind >= 0:
                    reward[i] += -1 if int(src_seq[ind]) != i + 1 else 1
                    if i > 0:
                        print(reward)
        if ind >= 0:
            src_seq[ind-2:ind+1] = 0
        dopamine_change = np.sum(0.2 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.001
        return dopamine_change

    duration = 30000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 9,
        "sigma": 0.2,
        "w_min": -16,
        "w_max": 16,
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.02,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.1,
        "tau_c": 100,
        "tau_d": 20
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 500
    initial_dopamine = 0.1
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params, plot_change=True)


def trial3():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        nonz = np.nonzero(src_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t:
                if ind >= 0:
                    reward[i] += -1 if int(src_seq[ind]) != i + 1 else 1
                    if i > 0:
                        print(reward)
        if ind >= 0:
            src_seq[ind] = 0
        dopamine_change = np.sum(0.02 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.001
        return dopamine_change

    duration = 30000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 8,
        "sigma": 0.2,
        "w_min": -16,
        "w_max": 16,
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.02,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.05,
        "tau_c": 100,
        "tau_d": 50
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 500
    initial_dopamine = 0.05
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params, regularize=(0.2, 0.1), plot_change=True)


def trial4():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        nonz = np.nonzero(src_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t:
                if ind >= 0:
                    reward[i] += -2 if int(src_seq[ind]) != i + 1 else 1
                    if i > 0:
                        print(reward)
        if ind >= 0:
            src_seq[ind-2:ind+1] = 0
        dopamine_change = np.sum(0.8 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.001
        return dopamine_change

    duration = 30000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 6,
        "sigma": 0.2,
        "w_min": -16,
        "w_max": 16,
        "a_plus": lambda x: 0.1,
        "a_minus": lambda x: -0.05,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.05,
        "tau_c": 100,
        "tau_d": 50
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 500
    initial_dopamine = 0.01
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params, plot_change=True)


def trial5():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        nonz = np.nonzero(src_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t:
                if ind >= 0:
                    reward[i] += -1 if int(src_seq[ind]) != i + 1 else 1
                    if i > 0:
                        print(reward)
        if ind >= 0:
            src_seq[ind] = 0
        dopamine_change = np.sum(0.2 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.0001
        return dopamine_change

    duration = 30000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 12,
        "sigma": 0.1,
        "w_min": -16,
        "w_max": 16,
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.01,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.05,
        "tau_c": 100,
        "tau_d": 50
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 500
    initial_dopamine = 0.05
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params, plot_change=True)


def trial6():
    def func_da(src_seq, dest_neurons, t):
        reward = [0 for _ in range(len(dest_neurons))]
        nonz = np.nonzero(src_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        for i in range(len(dest_neurons)):
            if len(dest_neurons[i].spike_times) > 0 and dest_neurons[i].spike_times[-1] == t:
                if ind >= 0:
                    reward[i] += -1 if int(src_seq[ind]) != i + 1 else 1
                    if i > 0:
                        print(reward)
        if ind >= 0:
            src_seq[ind] = 0
        dopamine_change = np.sum(0.2 * np.array(reward))
        if np.prod(np.array(reward)) < 0:
            dopamine_change -= 0.0001
        return dopamine_change

    duration = 30000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "full",
        "mu": 12,
        "sigma": 0.1,
        "w_min": -16,
        "w_max": 16,
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.01,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.05,
        "tau_c": 100,
        "tau_d": 50
    }

    src_size = 10
    dest_size = 2

    input_pattern = [
        [1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 3, 1, 1]
    ]

    interval = 500
    initial_dopamine = 0.05
    main(duration, dt, src_size, dest_size, interval, initial_dopamine, func_da, input_pattern,
         neuron_params, rstdp_params, plot_change=True)
