import numpy as np


class Population:
    def __init__(self, size, neuron_type, exc_ratio=1, trace_alpha=0.5, **neuron_params):
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

    def add(self, size, neuron):
        for i in range(size):
            self.neurons.append(neuron)
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
    def __init__(self, size, neuron_type, exc_ratio=1, trace_alpha=0.5, **neuron_params):
        super(InputPopulation, self).__init__(
            size, neuron_type, exc_ratio, trace_alpha, **neuron_params)
