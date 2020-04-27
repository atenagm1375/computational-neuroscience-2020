import numpy as np

from copy import deepcopy


class Connection:
    def __init__(self, pre, post, synapse):
        self.pre = pre
        self.post = post
        self.synapse = synapse

        self.pattern = [[]] * self.post.number

    def apply(self, connection_type, **kwargs):
        if connection_type == "full":
            self.synapse.w = self.synapse.w / self.post.number
            for i in range(self.pre.number):
                for j in range(self.post.number):
                    self.pattern[j].append([i, deepcopy(self.synapse)])

        elif connection_type == "fixed_prob":
            w, h = self.pre.number, self.post.number
            n = int(kwargs["p"] * h)
            self.synapse.w = self.synapse.w / n
            points = [divmod(i, h) for i in np.random.choice(range(w * h), n)]
            for x in points:
                self.pattern[x[1]].append([x[0], deepcopy(self.synapse)])

        elif connection_type == "fixed_pre":
            n = int(kwargs["p"] * self.post.number)
            for j in range(self.post.number):
                pres = np.random.choice(range(self.pre.number), n)
                for i in pres:
                    self.pattern[j].append([i, deepcopy(self.synapse)])

        else:
            raise ValueError("Invalid connection type!")

        return self

    def alpha(self, t, spikes):
        return [np.exp(f - t) for f in spikes]

    def post_input(self, t):
        for i in range(self.post.number):
            for j, syn in self.pattern[i]:
                self.post.neurons[i].input += syn.w * \
                    self.pre.neurons[j].effect(
                        self.alpha(t, self.pre.neurons[j].spike_times), t)
