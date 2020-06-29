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
    sns.scatterplot(
        x=spikes_per_neuron[:, 1], y=spikes_per_neuron[:, 0], s=20, hue=colors)
    plt.title("Raster Plot")
    plt.ylabel("Neuron")
    plt.xlabel("time")
    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()


def decision_plot(activity1, activity2, activity3=None, save_to=None):
    activity1 = np.array(activity1)
    activity2 = np.array(activity2)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80,
                     facecolor='w', edgecolor='k')
    plt.plot(activity1[:, 0], activity1[:, 1], 'b', label="Population 1")
    plt.plot(activity2[:, 0], activity2[:, 1], 'c', label="Population 2")
    if activity3:
        activity3 = np.array(activity3)
        plt.plot(activity3[:, 0], activity3[:, 1], 'r', label="Population 3")
    plt.title("POPULATION ACTIVITIES")
    plt.xlabel("time")
    plt.ylabel("Activity")
    plt.legend()
    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()


def plot_weight_matrix(weight_matrix):
    sns.heatmap(weight_matrix)
    plt.show()


def plot_weight_change(weight_change, save_to=None):
    weight_change = np.array(weight_change)
    a = weight_change.shape[1] * weight_change.shape[2]
    fig, axs = plt.subplots(2)
    x = np.array(weight_change)
    axs[0].plot(np.arange(len(weight_change)), x[:, :, 0])
    axs[0].set(xlabel='time', ylabel='weight')
    axs[0].set_title("Output Neuron 1")
    axs[1].plot(np.arange(len(weight_change)), x[:, :, 1])
    axs[1].set(xlabel='time', ylabel='weight')
    axs[1].set_title("Output Neuron 2")
    for ax in axs.flat:
        ax.label_outer()
    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()
    # plt.plot(np.arange(len(weight_change)), weight_change.reshape((weight_change.shape[0], a)))
    # plt.show()


def activity_plot(out_activities, save_to=None):
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80,
                     facecolor='w', edgecolor='k')

    # print(activity)
    for ind, activity in enumerate(out_activities):
        activity = np.array(activity)
        plt.plot(activity[:, 0],
                 activity[:, 1], label=f"output{ind + 1}")
    plt.title("Output units' activities")
    plt.xlabel("time")
    plt.ylabel("Activity")
    plt.legend()
    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()


def plot_images(images, titles=None, save_to=None):
    shape = images.shape

    if len(shape) == 3:
        shape = (1, shape[0], shape[1], shape[2])
    elif len(shape) < 2:
        shape = (1, shape[0])
    fig, axes = plt.subplots(shape[0], shape[1])
    if shape[0] == 1:
        if len(shape) == 4:
            axes = axes.reshape((1, shape[1]))
            images = images.reshape(shape)
            if titles is not None:
                titles = titles.reshape((1, shape[1]))
        else:
            axes = axes.reshape(shape)
            images = images.reshape(shape)
            if titles is not None:
                titles = titles.reshape(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            axes[i, j].imshow(images[i, j], "gray")
            if titles is not None:
                axes[i, j].set_title(titles[i, j])

    for ax in axes.flat:
        ax.label_outer()

    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()


def plot_images_time_to_spike(time_to_spikes, save_to=None):
    shape = time_to_spikes.shape
    if len(shape) > 2:
        shape = (shape[0], shape[1])
    fig, axes = plt.subplots(*shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            sns.heatmap(time_to_spikes[i, j], ax=axes[i, j])

    for ax in axes.flat:
        ax.label_outer()

    if save_to is None:
        plt.show()
    else:
        fig.savefig(save_to)
        plt.close()
