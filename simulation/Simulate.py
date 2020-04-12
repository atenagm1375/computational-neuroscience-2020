from Monitors import plot_firing_pattern


class Simulate:
    def __init__(self, neuron, time_step=1):
        self.dt = time_step
        self.__t = 0
        self.neuron = neuron

    def run(self, time_window, current):
        spike_times = []
        potential_list = []
        current_list = []
        time_list = []
        time_interval = range(self.__t, self.__t + time_window, self.dt)
        for t in time_interval:
            u, spiked = neuron.simulate(current, t, self.dt)
            potential_list.append(u)
            current_list.append(current(t))
            time_list.append(t)
            self.__t = t
            if spiked:
                spike_times.append(t)
        return potential_list, current_list, spike_times, time_list
