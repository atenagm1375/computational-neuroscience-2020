from models.Synapses import Synapse

import numpy as np

from copy import deepcopy


class Connection:
    def __init__(self, pre, post, initial_weights=10):
        self.pre = pre
        self.post = post
        self.initial_weights = initial_weights

    def apply(self, connection_type, **kwargs):
        if connection_type == "full":
            w = self.initial_weights / self.post.size
            for neuron_pre in self.pre.neurons:
                for neuron_post in self.post.neurons:
                    neuron_pre.target_synapses.append(
                        Synapse(neuron_pre, neuron_post, w))

        elif connection_type == "fixed_prob":
            w, h = self.pre.size, self.post.size
            n = int(kwargs["p"] * h)
            w = self.initial_weights / n
            points = [divmod(i, h) for i in np.random.choice(
                range(w * h), n, replace=False)]
            for x in points:
                self.pre[x[0]].target_synapses.append(
                    Synapse(self.pre.neurons[x[0]], self.post.neurons[x[1]], w))

        elif connection_type == "fixed_pre":
            n = int(kwargs["p"] * self.post.size)
            mu, sigma = kwargs["mu"], kwargs["sigma"]
            for neuron_post in self.post.neurons:
                pres = np.random.choice(range(self.pre.size), n, replace=False)
                for i in pres:
                    self.pre.neurons[i].target_synapses.append(
                        Synapse(self.pre.neurons[i], neuron_post, np.random.normal(mu, sigma)))

        else:
            raise ValueError("Invalid connection type!")

        return self
