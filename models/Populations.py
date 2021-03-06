import numpy as np


class Population:
    def __init__(self, size, neuron_type, input_part=None, output_part=None,
                 exc_ratio=1, trace_alpha=0.05, **neuron_params):
        self.size = size
        self.neurons = []
        self.input_part = input_part
        self.output_part = output_part
        self.exc_num = int(exc_ratio * size)
        self.inh_num = self.size - self.exc_num
        if self.input_part is not None:
            self.exc_num -= self.input_part.size
            for neuron in self.input_part.neurons:
                self.neurons.append(neuron)
        if self.output_part is not None:
            self.exc_num -= (len(self.output_part) * self.output_part[0].size)
            for out in self.output_part:
                for neuron in out.neurons:
                    self.neurons.append(neuron)
        for i in range(self.exc_num):
            self.neurons.append(neuron_type(**neuron_params))
        for i in range(self.inh_num):
            self.neurons.append(neuron_type(**neuron_params).set_inh())
        self.spikes_per_neuron = []
        self.trace_alpha = trace_alpha
        self.activity = []
        self.output_activity = None
        self.input_seq = []

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

    def encode_input(self, input_pattern, indices, duration, interval, dt=1):
        input_pattern = np.array(input_pattern)
        self.input_seq = -1 * np.ones(duration*dt)
        if input_pattern.shape[1] != indices.shape[1]:
            raise ValueError("Wrong input shape.")
        for t in np.arange(0, duration, interval):
            ind = np.random.choice(list(range(len(input_pattern))), 1)[0]
            if t + np.max(input_pattern[ind]) <= duration:
                for i, val in enumerate(input_pattern[ind]):
                    if val > 0 and t + val < duration:
                        neuron = self.neurons[indices[0][i]]
                        diff = (neuron.threshold - neuron.u_rest) / neuron.r
                        neuron.current_list[t + val] += (diff * neuron.tau)
                        self.input_seq[t + val] = ind

    def compute_potential(self, t, dt):
        for neuron in self.neurons:
            neuron.compute_potential(t, dt)

    def compute_spike(self, t, dt):
        self.activity.append([t, 0])
        if self.output_part is not None:
            for out_unit in self.output_part:
                out_unit.activity.append([t, 0])
        for neuron in self.neurons:
            neuron.compute_spike(t, dt)
            if len(neuron.spike_times) > 0 and neuron.spike_times[-1] == t:
                self.activity[-1][1] += 1
                if self.output_part is not None:
                    for out_unit in self.output_part:
                        if neuron in out_unit.neurons:
                            out_unit.activity[-1][1] += (1 / out_unit.size)
                            break
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


class InputPopulation(Population):
    def __init__(self, size, neuron_type, exc_ratio=1, **neuron_params):
        super(InputPopulation, self).__init__(
            size, neuron_type, exc_ratio=exc_ratio, trace_alpha=0, input_part=None, output_part=None, **neuron_params)

        # self.input = []

    def encode(self, input_pattern, duration, interval, dt=1):
        input_pattern = np.array(input_pattern)
        self.input_seq = np.zeros(duration*dt)
        # self.input_seq = list(self.input_seq)
        if input_pattern.shape[1] != self.size:
            raise ValueError("Wrong input shape.")
        # self.input = input_pattern
        for t in np.arange(0, duration, interval):
            ind = np.random.choice(list(range(len(input_pattern))), 1)[0]
            if t + np.max(input_pattern[ind]) <= duration:
                for i, val in enumerate(input_pattern[ind]):
                    neuron = self.neurons[i]
                    diff = (neuron.threshold - neuron.u_rest) / neuron.r
                    if val > 0:
                        neuron.current_list[t + val] += (diff * neuron.tau / dt)
                        self.input_seq[t + val] = ind + 1
