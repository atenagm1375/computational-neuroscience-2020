class Population:
    def __init__(self, number, neuron):
        self.number = number
        self.neurons = [neuron] * number
        self.spikes_per_neuron = [neuron.spike_times] * number
        self.connection_pattern = dict(
            zip([i for i in range(number)], [[]] * number))

    def add(self, number, neuron):
        for i in range(number):
            self.neurons.append(neuron)
            self.connection_pattern[i + self.number] = []
            self.spikes_per_neuron.append(neuron.spike_times)
        self.number += number
