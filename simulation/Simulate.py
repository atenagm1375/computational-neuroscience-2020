import numpy as np


class Network:
    def __init__(self, neuron=None, synapse=None, populations=None, connections=None, time_step=1):
        self.dt = time_step
        self.__t = 0
        self.neuron = neuron
        self.synapse = synapse
        self.populations = populations
        self.connections = connections

        self.in_unit = self.populations[0]
        self.out_unit = self.populations[-1]
        self.d = 0
        self.func_da = None

    def set_input_output_neurons(self, in_indices, out_indices):
        self.in_unit = in_indices[0]
        self.out_unit = [np.array(self.populations[0].neurons)[out.astype(int)] for out in out_indices]

    def set_dopamine(self, d, da):
        self.d = d
        self.func_da = da

    def da(self, t):
        if len(self.populations) > 1:
            return self.func_da(self.populations[0].input_seq, self.populations[-1].neurons, t)
        return self.func_da(self.populations[0], t)

    def run(self, time_window, learning_rule=None):
        if self.neuron:
            current_list = []
            time_list = []
            time_interval = np.arange(
                self.__t, self.__t + time_window, self.dt)
            for t in time_interval:
                self.neuron.step(t, self.dt)
                # self.neuron.compute_potential(t, self.dt)
                # self.neuron.compute_spike(t, self.dt)
                current_list.append(self.neuron.current(int(t // self.dt)))
                time_list.append(t)
                if len(self.neuron.spike_times) > 0 and self.neuron.spike_times[-1] == t:
                    current_list.append(
                        self.neuron.current(int(t // self.dt)))
                    time_list.append(t)
                self.__t = t
        else:
            current_list = []
            time_list = []
            time_interval = np.arange(
                self.__t, self.__t + time_window, self.dt)
            for pop in self.populations:
                for neuron in pop.neurons:
                    neuron.input = np.zeros(time_window // self.dt)
                    neuron.duration = time_window
            for t in time_interval:
                if t % 10 == 0:
                    print(t)
                for pop in self.populations:
                    pop.compute_potential(t, self.dt)
                for pop in self.populations:
                    pop.apply_pre_synaptic(t, self.dt)
                for pop in self.populations:
                    pop.compute_spike(t, self.dt)
                for pop in self.populations:
                    pop.reset(t, self.dt)
                for conn in self.connections:
                    up = conn.update(learning_rule, t, self.dt, self.d, self.da)
                    if learning_rule == "rstdp" and up is not None:
                        self.d = up[-1]
                # for conn in self.connections:
                #     for syn in conn.synapses:
                #         print(syn.pre.name, "--{}-->".format(syn.w), syn.post.name)
                # current_list.append(pop.neurons[0]._current(int(t // self.dt)))
                time_list.append(t)
            for pop in self.populations:
                pop.compute_spike_history()

        return current_list, time_list
