import matplotlib.pyplot as plt


def plot_f_i_curve(firing_patterns, time_window, currents):
    frequencies = [len(spikes) / time_window for spikes in firing_patterns]
    plt.plot(frequencies, currents, 'r')
    plt.set(xlabel='I(t)', ylabel='f = 1/T')
    plt.set_title("frequency-current relation")
    plt.show()


def plot_firing_pattern(potentials, currents, times, u_rest, threshold):
    fig, axs = plt.subplots(2)
    axs[0].plot(potentials, times, 'g-')
    axs[0].plot(u_rest, times, 'k--')
    axs[0].plot(threshold, times, 'b--')
    axs[0].set(xlabel='time', ylabel='u(t)')
    axs[0].legend()
    axs[1].plot(currents, times, 'b-')
    axs[1].set(xlabel='time', ylabel='I(t)')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
