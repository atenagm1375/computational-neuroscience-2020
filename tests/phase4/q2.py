from models.Neurons import LIF
from models.Populations import Population, InputPopulation
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


def main(trial_no, duration, dt, population_size, n, input_size, output_size, interval, initial_dopamine,
         func_da, neuron_params, rstdp_params, output_current=None):

    inp = InputPopulation(input_size, LIF, **neuron_params)
    input_pattern = []
    for i in range(n):
        input_pattern.append(np.random.randint(0, 5, input_size))
    inp.encode(input_pattern, duration, interval, dt)
    if output_current is not None:
        neuron_params["current"] = output_current * np.ones(duration // dt)
    outs = []
    # neuron_params["regularize"] = True
    for i in range(n):
        outs.append(Population(output_size, LIF, trace_alpha=0, **neuron_params))
    pop = Population(population_size, LIF, input_part=inp, output_part=outs, exc_ratio=0.8,
                     trace_alpha=0, **neuron_params)

    conn = Connection(pop, pop, weight_change=False).apply(**rstdp_params)

    net = Network(populations=[pop], connections=[conn], time_step=dt)
    net.set_dopamine(initial_dopamine, func_da)
    net.run(duration, learning_rule="rstdp")
    inp.compute_spike_history()
    raster_plot(np.array(inp.spikes_per_neuron))
    activity_plot([out.activity for out in outs], save_to="./q2trial{}.png".format(trial_no))


def trial1():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(1, duration, dt, 100, 2, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial2():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(2, duration, dt, 100, 5, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial3():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(3, duration, dt, 100, 10, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial4():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(4, duration, dt, 100, 2, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial5():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(5, duration, dt, 100, 5, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial6():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(6, duration, dt, 100, 10, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial7():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(7, duration, dt, 100, 2, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial8():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(8, duration, dt, 100, 5, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial9():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.02 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.00001
        return dopamine_change

    duration = 25000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 1,
        "threshold": -65,
        "current": current_generator(duration, dt, 1, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 15,
        "sigma": 0.2,
        "w_min": -20,
        "w_max": 20,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.005,
        "tau_plus": 4,
        "tau_minus": 4,
        "c": 0.01,
        "tau_c": 100,
        "tau_d": 50
    }

    main(9, duration, dt, 100, 10, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)


def trial10():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.2 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.005
        return dopamine_change

    duration = 5000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 2,
        "threshold": -65,
        "current": current_generator(duration, dt, 0.5, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 0.8,
        "sigma": 0.1,
        "w_min": -5,
        "w_max": 5,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.1,
        "a_minus": lambda x: -0.05,
        "tau_plus": 4,
        "tau_minus": 3,
        "c": 0.1,
        "tau_c": 50,
        "tau_d": 15
    }

    main(10, duration, dt, 1000, 2, 50, 20, 100, 0.1, func_da, neuron_params, rstdp_params, output_current=4)


def trial11():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.2 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.005
        return dopamine_change

    duration = 5000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 2,
        "threshold": -65,
        "current": current_generator(duration, dt, 0.5, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 0.8,
        "sigma": 0.1,
        "w_min": -5,
        "w_max": 5,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.1,
        "a_minus": lambda x: -0.05,
        "tau_plus": 4,
        "tau_minus": 3,
        "c": 0.1,
        "tau_c": 50,
        "tau_d": 15
    }

    main(11, duration, dt, 1000, 5, 50, 20, 100, 0.1, func_da, neuron_params, rstdp_params, output_current=4)


def trial12():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.2 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.01
        return dopamine_change

    duration = 5000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 2,
        "threshold": -65,
        "current": current_generator(duration, dt, 0.5, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.25,
        "mu": 0.5,
        "sigma": 0.1,
        "w_min": -5,
        "w_max": 5,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.1,
        "a_minus": lambda x: -0.05,
        "tau_plus": 4,
        "tau_minus": 3,
        "c": 0.1,
        "tau_c": 50,
        "tau_d": 15
    }

    main(12, duration, dt, 1000, 2, 50, 20, 100, 0.1, func_da, neuron_params, rstdp_params, output_current=4)


def trial13():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        activities = [out.activity[-1][1] for out in pop.output_part]
        most_active = np.argmax(activities)
        nonz = np.nonzero(pop.input_part.input_seq[:t])[0]
        if nonz.size > 0:
            ind = np.max(nonz)
        else:
            ind = -1
        if ind >= 0:
            reward[int(most_active)] = 1 if pop.input_part.input_seq[ind] == most_active else -1

        dopamine_change = np.sum(0.2 * np.array(reward))
        if activities.count(most_active) > 1:
            dopamine_change -= 0.005
        return dopamine_change

    duration = 5000
    dt = 1

    neuron_params = {
        "tau": 5,
        "r": 2,
        "threshold": -65,
        "current": current_generator(duration, dt, 0.5, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.25,
        "mu": 0.5,
        "sigma": 0.1,
        "w_min": -5,
        "w_max": 5,
        "delay": np.random.randint(1, 5, 1),
        "a_plus": lambda x: 0.1,
        "a_minus": lambda x: -0.1,
        "tau_plus": 4,
        "tau_minus": 3,
        "c": 0.1,
        "tau_c": 50,
        "tau_d": 15
    }

    main(13, duration, dt, 1000, 5, 50, 20, 100, 0.1, func_da, neuron_params, rstdp_params, output_current=3.5)
