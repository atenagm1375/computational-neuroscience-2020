import matplotlib.pyplot as plt


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
        plt.savefig(save_to)
