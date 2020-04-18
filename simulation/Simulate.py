import numpy as np


class Simulate:
    def __init__(self, neuron, time_step=1):
        self.dt = time_step
        self.__t = 0
        self.neuron = neuron

    def run(self, time_window, current):
        potential_list = []
        current_list = []
        time_list = []
        time_interval = np.arange(self.__t, self.__t + time_window, self.dt)
        for t in time_interval:
            self.neuron._simulate(current, t, self.dt)
            current_list.append(current(t))
            time_list.append(t)
            if t in self.neuron.spike_times:
                current_list.append(current(t))
                time_list.append(t)
            self.__t = t
        return self.neuron.potential_list, current_list, time_list
