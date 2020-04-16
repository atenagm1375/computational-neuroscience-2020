from models.Neurons import LIF, ELIF, AddaptiveELIF
from simulation.Monitors import plot_f_i_curve, plot_firing_pattern
from simulation.Simulate import Simulate
import numpy as np

import pickle
import sys


def constant_current_test(Neuron, current_values, time_window, dt, **neuron_params):
    firing_patterns = []
    for current_value in current_values:
        lif = Neuron(**neuron_params)
        simulate = Simulate(lif, dt)
        potential_list, current_list, time_list = simulate.run(
            time_window, lambda x: current_value)
        firing_patterns.append(lif.spike_times)
        plot_firing_pattern(potential_list, current_list,
                            time_list, lif.u_rest, lif.threshold)
    plot_f_i_curve(firing_patterns, time_window, current_values)


def random_current_test(Neuron, current_range, time_window, dt, **neuron_params):
    lif = Neuron(**neuron_params)
    simulate = Simulate(lif, dt)
    potential_list, current_list, time_list = simulate.run(
        time_window, lambda x: np.random.randint(*current_range))
    plot_firing_pattern(potential_list, current_list,
                        time_list, lif.u_rest, lif.threshold)


def question1(parameter_sets):
    for param_set in parameter_sets:
        print(param_set)
        constant_current_test(
            LIF, param_set['current_values'], param_set['time_window'],
            param_set['dt'], **param_set['neuron_params'])


def question2(parameter_sets):
    for param_set in parameter_sets:
        print(param_set)
        random_current_test(
            LIF, param_set['current_range'], param_set['time_window'],
            param_set['dt'], **param_set['neuron_params'])


def question3(parameter_sets):
    for param_set in parameter_sets:
        print(param_set)
        constant_current_test(
            ELIF, param_set['current_values'], param_set['time_window'],
            param_set['dt'], **param_set['neuron_params'])

    for param_set in parameter_sets:
        print(param_set)
        random_current_test(
            ELIF, param_set['current_range'], param_set['time_window'],
            param_set['dt'], **param_set['neuron_params'])


def question4(parameter_sets):
    for param_set in parameter_sets:
        print(param_set)
        constant_current_test(
            AddaptiveELIF, param_set['current_values'], param_set['time_window'],
            param_set['dt'], **param_set['neuron_params'])

    for param_set in parameter_sets:
        print(param_set)
        random_current_test(
            AddaptiveELIF, param_set['current_range'], param_set['time_window'],
            param_set['dt'], **param_set['neuron_params'])


if __name__ == "__main__":
    with open(f'./tests/phase1/{sys.argv[2]}', 'rb') as file:
        globals()["question{}".format(sys.argv[1])](pickle.load(file))
