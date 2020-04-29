import numpy as np


class Synapse:
    def __init__(self, pre, post, weight, delay=0):
        self.pre = pre
        self.post = post
        self.w = weight
        self.d = delay

    def alpha_rate(self, t):
        return np.sum([1 - np.exp(f - t) for f in self.pre.spike_times])
