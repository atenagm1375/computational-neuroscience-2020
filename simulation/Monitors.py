import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


def plot_f_i_curve(firing_patterns, time_window, currents, save_to=None):
    frequencies = [len(spikes) / time_window for spikes in firing_patterns]
    plt.plot(currents, frequencies, 'r')
    plt.xlabel(xlabel='I(t)')
    plt.ylabel(ylabel='f = 1/T')
    plt.title("frequency-current relation")
    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)
        plt.close()


def plot_firing_pattern(potentials, currents, times, u_rest, threshold, save_to=None):
    fig, axs = plt.subplots(2)
    axs[0].plot(times, potentials, 'g-')
    axs[0].plot(times, [u_rest] * len(times), 'k--')
    axs[0].plot(times, [threshold] * len(times), 'b--')
    axs[0].set(xlabel='time', ylabel='u(t)')
    axs[1].plot(times, currents, 'b-')
    axs[1].set(xlabel='time', ylabel='I(t)')
    for ax in axs.flat:
        ax.label_outer()
    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()


def raster_plot(spikes_per_neuron, colors=None, save_to=None):
    # plt.eventplot(spikes_per_neuron, color=colors, linelengths=0.05)
    if spikes_per_neuron.shape[0] == 0:
        print("NO ACTIVITY")
        return
    # plt.scatter(x=spikes_per_neuron[:, 1],
    #             y=spikes_per_neuron[:, 0], marker='.')
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80,
                     facecolor='w', edgecolor='k')
    sns.scatterplot(x=spikes_per_neuron[:, 1], y=spikes_per_neuron[:, 0])
    plt.title("Raster Plot")
    plt.ylabel("Neuron")
    plt.xlabel("time")
    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()


def decision_plot(activity1, activity2, save_to=None):
    activity1 = np.array(activity1)
    activity2 = np.array(activity2)
    plt.plot(activity1[:, 0], activity1[:, 1], 'b', label="Population 1")
    plt.plot(activity2[:, 0], activity2[:, 1], 'c', label="Population 2")
    plt.title("POPULATION ACTIVITIES")
    plt.xlabel("time")
    plt.ylabel("Activity")
    plt.legend()
    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()
