class Population:
    def __init__(self, number, neuron):
        self.number = number
        self.neurons = [neuron] * number
        self.spikes_per_neuron = [neuron.spike_times] * number

    def add(self, number, neuron):
        for i in number:
            self.neurons.append(self.neuron)
        self.number += number

    def _simulate(self, current, t, dt):
        for i, neuron in enumerate(self.neurons):
            neuron._simulate(current, t, dt)
            self.spikes_per_neuron[i] = neuron.spike_times
