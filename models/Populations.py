from abc import ABC
import numpy as np

from models.Neurons import Input


class Population:
    def __init__(self, size, neuron_type, exc_ratio=1, trace_alpha=2, **neuron_params):
        self.size = size
        exc_num = int(exc_ratio * size)
        self.neurons = []
        exc_num = int(exc_ratio * size)
        for i in range(exc_num):
            self.neurons.append(neuron_type(**neuron_params))
        for i in range(self.size - exc_num):
            self.neurons.append(neuron_type(**neuron_params)._set_inh())
        self.spikes_per_neuron = []
        self.trace_alpha = trace_alpha
        self.activity = []

    def add(self, size, neuron_type, **neuron_params):
        for i in range(size):
            self.neurons.append(neuron_type(**neuron_params))
        self.size += size

    def compute_potential(self, t, dt):
        for neuron in self.neurons:
            neuron.compute_potential(t, dt)

    def compute_spike(self, t, dt):
        self.activity.append([t, 0])
        for neuron in self.neurons:
            neuron.compute_spike(t, dt)
            if len(neuron.spike_times) > 0 and neuron.spike_times[-1] == t:
                self.activity[-1][1] += 1
        self.activity[-1][1] /= self.size

    def apply_pre_synaptic(self, t, dt):
        for neuron in self.neurons:
            neuron.apply_pre_synaptic(t, dt)

    # def step(self, t, dt):
    #     self.activity.append([t, 0])
    #     for neuron in self.neurons:
    #         neuron.step(t, dt)
    #         if len(neuron.spike_times) > 0 and neuron.spike_times[-1] == t:
    #             self.activity[-1][1] += 1
    #     self.activity[-1][1] /= self.size

    def input_reset(self, t, dt):
        for neuron in self.neurons:
            neuron.input_reset(t, dt, self.trace_alpha)

    def compute_spike_history(self):
        spike_history = []
        for i, neuron in enumerate(self.neurons):
            for t_f in neuron.spike_times:
                spike_history.append([i, t_f])
        self.spikes_per_neuron = np.array(spike_history)


class InputPopulation(Population):
    def __init__(self, size, exc_ratio=1, trace_alpha=2, interval=1):
        super(InputPopulation, self).__init__(
            size, Input, exc_ratio, trace_alpha, interval=interval)

        self.interval = interval
        self.input = []

    def set_input(self, input):
        input = np.array(input)
        if input.shape[1] != self.size:
            raise ValueError("Wrong input shape.")
        self.input = input
        for i, neuron in enumerate(self.neurons):
            neuron.set(input[:, i])

    def compute_potential(self, t, dt):
        if t % self.interval == 0:
            values = np.random.choice(list(range(len(self.input))), 1)[0]
        for i, neuron in enumerate(self.neurons):
            if t % self.interval == 0 and self.input[values][i] != 0:
                neuron.input.append(t + self.input[values][i])
            neuron.compute_potential(t, dt)

    def compute_spike(self, t, dt):
        for neuron in self.neurons:
            neuron.compute_spike(t, dt)

    def apply_pre_synaptic(self, t, dt):
        pass

    def input_reset(self, t, dt):
        pass


class InputPopulation2(Population):
    def __init__(self, size, neuron_type, exc_ratio=1, **neuron_params):
        super(InputPopulation2, self).__init__(
            size, neuron_type, exc_ratio, trace_alpha=0, **neuron_params)

        self.input = []

    def encode(self, input, duration, interval):
        input = np.array(input)
        if input.shape[1] != self.size:
            raise ValueError("Wrong input shape.")
        self.input = input
        for t in np.arange(0, duration, interval):
            ind = np.random.choice(list(range(len(self.input))), 1)[0]
            if t + np.max(self.input[ind]) <= duration:
                for i, val in enumerate(self.input[ind]):
                    if val > 0:
                        self.neurons[i].current_list[t + val] += 1
