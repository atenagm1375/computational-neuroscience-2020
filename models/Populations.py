from abc import ABC
import numpy as np

from models.Neurons import Input


class Population:
    def __init__(self, size, neuron_type, exc_ratio=1, trace_alpha=2, **neuron_params):
        self.size = size
        self.neurons = []
        self.exc_num = int(exc_ratio * size)
        for i in range(self.exc_num):
            self.neurons.append(neuron_type(**neuron_params))
        for i in range(self.size - self.exc_num):
            self.neurons.append(neuron_type(**neuron_params)._set_inh())
        self.spikes_per_neuron = []
        self.trace_alpha = trace_alpha
        self.activity = []

    def add(self, size, neuron_type, **neuron_params):
        for i in range(size):
            self.neurons.append(neuron_type(**neuron_params))
        self.size += size

    def choose_input_output(self, input_size, output_size):
        distinct_choices = np.random.choice(list(range(self.exc_num)),
                                            np.prod(input_size) +
                                            np.prod(output_size),
                                            replace=False)
        input_indices = []
        for i in range(input_size[0]):
            ins = np.random.choice(
                distinct_choices, input_size[1], replace=False)
            input_indices.append(ins)
            distinct_choices = list(set(distinct_choices) - set(ins))

        output_indices = []
        for i in range(output_size[0]):
            ins = np.random.choice(
                distinct_choices, output_size[1], replace=False)
            output_indices.append(ins)
            distinct_choices = list(set(distinct_choices) - set(ins))
        return input_indices, output_indices

    def encode_input(self, input_pattern, indices, duration, interval):
        input_pattern = np.array(input_pattern)
        if input_pattern.shape[1] != indices.shape[1]:
            raise ValueError("Wrong input shape.")
        for t in np.arange(0, duration, interval):
            ind = np.random.choice(list(range(len(input_pattern))), 1)[0]
            if t + np.max(input_pattern[ind]) <= duration:
                for i, val in enumerate(input_pattern[ind]):
                    if val > 0:
                        neuron = self.neurons[indices[i]]
                        diff = (neuron.threshold - neuron.u_rest) / neuron.r
                        neuron.current_list[t + val] += (diff * neuron.tau)

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

    def reset(self, t, dt):
        for neuron in self.neurons:
            neuron.reset(t, dt, self.trace_alpha)

    def compute_spike_history(self):
        spike_history = []
        for i, neuron in enumerate(self.neurons):
            for t_f in neuron.spike_times:
                spike_history.append([i, t_f])
        self.spikes_per_neuron = np.array(spike_history)


# class InputPopulation(Population):
#     def __init__(self, size, exc_ratio=1, trace_alpha=2, interval=1):
#         super(InputPopulation, self).__init__(
#             size, Input, exc_ratio, trace_alpha, interval=interval)
#
#         self.interval = interval
#         self.input = []
#
#     def set_input(self, input):
#         input = np.array(input)
#         if input.shape[1] != self.size:
#             raise ValueError("Wrong input shape.")
#         self.input = input
#         for i, neuron in enumerate(self.neurons):
#             neuron.set(input[:, i])
#
#     def compute_potential(self, t, dt):
#         if t % self.interval == 0:
#             values = np.random.choice(list(range(len(self.input))), 1)[0]
#         for i, neuron in enumerate(self.neurons):
#             if t % self.interval == 0 and self.input[values][i] != 0:
#                 neuron.input.append(t + self.input[values][i])
#             neuron.compute_potential(t, dt)
#
#     def compute_spike(self, t, dt):
#         for neuron in self.neurons:
#             neuron.compute_spike(t, dt)
#
#     def apply_pre_synaptic(self, t, dt):
#         pass
#
#     def reset(self, t, dt):
#         pass


class InputPopulation2(Population):
    def __init__(self, size, neuron_type, exc_ratio=1, **neuron_params):
        super(InputPopulation2, self).__init__(
            size, neuron_type, exc_ratio, trace_alpha=0, **neuron_params)

        # self.input = []

    def encode(self, input_pattern, duration, interval):
        input_pattern = np.array(input_pattern)
        if input_pattern.shape[1] != self.size:
            raise ValueError("Wrong input shape.")
        # self.input = input_pattern
        for t in np.arange(0, duration, interval):
            ind = np.random.choice(list(range(len(input_pattern))), 1)[0]
            if t + np.max(input_pattern[ind]) <= duration:
                for i, val in enumerate(input_pattern[ind]):
                    if val > 0:
                        neuron = self.neurons[i]
                        diff = (neuron.threshold - neuron.u_rest) / neuron.r
                        neuron.current_list[t + val] += (diff * neuron.tau)
