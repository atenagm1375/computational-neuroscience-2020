from models.Neurons import LIF
from models.Populations import Population, InputPopulation
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

    inp = InputPopulation(input_size, LIF, **neuron_params)
    input_pattern = []
    for i in range(n):
        input_pattern.append(np.random.randint(0, 3, input_size))
    inp.encode(input_pattern, duration, interval, dt)
    neuron_params["current"] = np.zeros(duration // dt)
    outs = []
    # neuron_params["regularize"] = True
    for i in range(n):
        outs.append(Population(output_size, LIF, trace_alpha=0, **neuron_params))
    pop = Population(population_size, LIF, input_part=inp, output_part=outs, exc_ratio=0.8,
                     trace_alpha=0.5, **neuron_params)

    conn = Connection(pop, pop, weight_change=False).apply(**rstdp_params)

    net = Network(populations=[pop], connections=[conn], time_step=dt)
    net.set_dopamine(initial_dopamine, func_da)
    net.run(duration, learning_rule="rstdp")
    # inp.compute_spike_history()
    # raster_plot(np.array(inp.spikes_per_neuron))
    activity_plot([out.activity for out in outs])


def trial1():
    def func_da(pop, t):
        reward = [0 for _ in range(len(pop.output_part))]
        most_active = np.argmax([out.activity[-1][1] for out in pop.output_part])
        reward[int(most_active)] = 1 if pop.input_part.input_seq[t] == most_active else -2
        # for i in range(len(reward)):
        #     if i != most_active:
        #         reward[i] = -1 / (len(pop.output_part) - 1)

        # print(reward)
        dopamine_change = np.sum(1.25 * np.array(reward))
        return dopamine_change

    duration = 50000
    dt = 1

    neuron_params = {
        "tau": 10,
        "r": 1,
        "threshold": -55,
        "current": current_generator(duration, dt, 0.5, 5, 1)
    }

    rstdp_params = {
        "connection_type": "fixed_pre",
        "p": 0.1,
        "mu": 0.6,
        "sigma": 0.05,
        "delay": np.random.randint(1, 10, 1),
        "a_plus": lambda x: 0.01,
        "a_minus": lambda x: -0.01,
        "tau_plus": 8,
        "tau_minus": 8,
        "c": 0.5,
        "tau_c": 100,
        "tau_d": 50
    }

    main(duration, dt, 1000, 2, 5, 2, 500, 0.01, func_da, neuron_params, rstdp_params)
