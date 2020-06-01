from models.Synapses import Synapse

import numpy as np

from copy import deepcopy


class Connection:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post
        self.synapses = []

    def add(self, pre_indices, post_indices, mu=0.5, sigma=0.01, **kwargs):
        for i in pre_indices:
            for j in post_indices:
                syn = Synapse(self.pre.neurons[i], self.post.neurons[j],
                              np.random.normal(mu, sigma), **kwargs)
                self.synapses.append(syn)
                self.pre.neurons[i].target_synapses.append(syn)
        return self

    def apply(self, connection_type, mu=0.5, sigma=0.01, **kwargs):
        if connection_type == "full":
            for neuron_pre in self.pre.neurons:
                for neuron_post in self.post.neurons:
                    syn = Synapse(neuron_pre, neuron_post,
                                  np.random.normal(mu, sigma), **kwargs)
                    self.synapses.append(syn)
                    neuron_pre.target_synapses.append(syn)

        elif connection_type == "fixed_prob":
            w, h = self.pre.size, self.post.size
            n = int(kwargs["p"] * h * w)
            points = [divmod(i, h) for i in np.random.choice(
                list(range(w * h)), n, replace=False)]
            for x in points:
                syn = Synapse(self.pre.neurons[x[0]], self.post.neurons[x[1]],
                              np.random.normal(mu, sigma), **kwargs)
                self.synapses.append(syn)
                self.pre.neurons[x[0]].target_synapses.append(syn)

        elif connection_type == "fixed_pre":
            n = int(kwargs["p"] * self.post.size)
            for neuron_post in self.post.neurons:
                pres = np.random.choice(range(self.pre.size), n, replace=False)
                for i in pres:
                    syn = Synapse(self.pre.neurons[i], neuron_post,
                                  np.random.normal(mu, sigma), **kwargs)
                    self.synapses.append(syn)
                    self.pre.neurons[i].target_synapses.append(syn)

        else:
            raise ValueError("Invalid connection type!")

        return self

    def update(self, learning_rule, t, dt, d=0, da=None):
        if learning_rule:
            for synapse in self.synapses:
                synapse.update(learning_rule, t, dt, d, da)
