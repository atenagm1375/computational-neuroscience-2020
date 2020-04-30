import models

import numpy as np


class Simulate:
    def __init__(self, neuron=None, synapse=None, populations=None, connections=None, time_step=1):
        self.dt = time_step
        self.__t = 0
        self.neuron = neuron
        self.synapse = synapse
        self.populations = populations
        self.connections = connections

    def run(self, time_window):
        if self.neuron:
            current_list = []
            time_list = []
            time_interval = np.arange(
                self.__t, self.__t + time_window, self.dt)
            for t in time_interval:
                self.neuron.step(t, self.dt)
                current_list.append(self.neuron._current(int(t // self.dt)))
                time_list.append(t)
                if len(self.neuron.spike_times) > 0 and self.neuron.spike_times[-1] == t:
                    current_list.append(
                        self.neuron._current(int(t // self.dt)))
                    time_list.append(t)
                self.__t = t
        else:
            current_list = []
            time_list = []
            time_interval = np.arange(
                self.__t, self.__t + time_window, self.dt)
            for t in time_interval:
                if t % 10 == 0:
                    print(t)
                for pop in self.populations:
                    pop.step(t, self.dt)
                for pop in self.populations:
                    pop.input_reset(t)
                current_list.append(pop.neurons[0]._current(int(t // self.dt)))
                time_list.append(t)
            for pop in self.populations:
                pop.compute_spike_history()

        return current_list, time_list
