from models.Neurons import LIF, ELIF, AddaptiveELIF
from simulation.Monitors import plot_f_i_curve, plot_firing_pattern
from simulation.Simulate import Network
import numpy as np

import pickle
import sys


def current_generator(duration, dt, values):
    current_list = []
    for i in np.arange(duration * dt):
        if i < 5:
            current_list.append(0)
        elif i < duration - 5:
            if isinstance(values, tuple):
                current_list.append(np.random.uniform(*values))
            else:
                current_list.append(values)
        else:
            current_list.append(0)
    return current_list


def constant_current_test(Neuron, current_values, time_window, dt, **neuron_params):
    firing_patterns = []
    f_i_curve_name = f"./tests/phase1/gain_function_{neuron_params}.png"
    for current_value in current_values:
        plot_name = f"./tests/phase1/{neuron_params}_{time_window}_{current_value}.png"
        lif = Neuron(current=current_generator(
            time_window, dt, current_value), **neuron_params)
        simulate = Network(lif, dt)
        current_list, time_list = simulate.run(time_window)
        firing_patterns.append(lif.spike_times)
        plot_firing_pattern(lif.potential_list, current_list,
                            time_list, lif.u_rest, lif.threshold, save_to=plot_name)
    plot_f_i_curve(firing_patterns, time_window,
                   current_values, save_to=f_i_curve_name)


def random_current_test(Neuron, current_range, time_window, dt, **neuron_params):
    plot_name = f"./tests/phase1/{neuron_params}_{time_window}_randomCurrent.png"
    lif = Neuron(current=current_generator(
        time_window, dt, current_range), **neuron_params)
    simulate = Network(lif, dt)
    current_list, time_list = simulate.run(time_window)
    plot_firing_pattern(lif.potential_list, current_list,
                        time_list, lif.u_rest, lif.threshold, save_to=plot_name)


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
    with open(f'./tests/phase1/q{sys.argv[1]}.data', 'rb') as file:
        globals()["question{}".format(sys.argv[1])](pickle.load(file))
