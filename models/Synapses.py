import numpy as np


class Synapse:
    def __init__(self, pre, post, weight, delay=1, **parameters):
        self.pre = pre
        self.post = post
        self.w = weight
        self.d = delay
        self.parameters = parameters

    def _stdp(self, t, t_pre, t_post):
        a_plus = self.parameters["a_plus"]
        a_minus = self.parameters["a_minus"]
        tau_plus = self.parameters["tau_plus"]
        tau_minus = self.parameters["tau_minus"]
        delta_t = np.abs(t_post - t_pre)
        dw_plus, dw_minus = 0, 0
        if t == t_post:
            dw_plus = a_plus(self.w) * np.exp(-delta_t / tau_plus)
        if t == t_pre:
            dw_minus = a_minus(self.w) * np.exp(-delta_t / tau_minus)
        return delta_t, dw_plus, dw_minus

    def stdp_rule(self, t):
        t_pre, t_post = -1, -1
        if pre_neuron.spike_times:
            t_pre = self.pre.spike_times[-1]
        if post_neuron.spike_times:
            t_post = self.post.spike_times[-1]
        if t_post >= 0 and t_pre >= 0:
            delta_t, dw_plus, dw_minus = _stdp(t, t_pre, t_post)
            self.w += (dw_plus + dw_minus)
            return delta_t, dw_plus + dw_minus

    def rstdp_rule(self, t):
        pass

    def plasticity(self, learning_rule, t):
        if learning_rule == "stdp":
            return stdp_rule(t)
        elif learning rule == "rstdp":
            return rstdp_rule(t)
        else:
            raise ValueError("INVALID LEARNING RULE")
