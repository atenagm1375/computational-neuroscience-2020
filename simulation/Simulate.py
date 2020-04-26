import models

import numpy as np


class Simulate:
    def __init__(self, components, time_step=1):
        self.dt = time_step
        self.__t = 0
        self.components = components

    def run(self, time_window, current):
        if isinstance(self.components, models.Neuron.LIF):
            current_list = []
            time_list = []
            time_interval = np.arange(
                self.__t, self.__t + time_window, self.dt)
            for t in time_interval:
                self.components._simulate(current(t), t, self.dt)
                current_list.append(current(t))
                time_list.append(t)
                if t in self.components.spike_times:
                    current_list.append(current(t))
                    time_list.append(t)
                self.__t = t
            return current_list, time_list
        else:
            for t in time_interval:
                for connection in self.components["connections"]:
                    for post_idx in connection.pattern:
                        connection.post[post_idx]._simulate(
                            current(t), t, self.dt)
                        for i, syn in post_idx:
                            connection.pre.neuron[i]._simulate(
                                current(t), t, self.dt)
                for connection in self.components["connections"]:
                    connection.post_input(t)
