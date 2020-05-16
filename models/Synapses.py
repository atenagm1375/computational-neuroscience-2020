import numpy as np


class Synapse:
    def __init__(self, pre, post, weight, delay=1, trace_alpha=5, **parameters):
        self.pre = pre
        self.post = post
        self.w = weight
        self.d = delay
        self.parameters = parameters
        self.trace_alpha = trace_alpha

    def _stdp(self, t, t_pre, t_post):
        a_plus = self.parameters["a_plus"]
        a_minus = self.parameters["a_minus"]
        tau_plus = self.parameters["tau_plus"]
        tau_minus = self.parameters["tau_minus"]
        delta_t = t_post - t_pre
        dw = 0
        if delta_t >= 0:
            dw = a_plus(self.w) * np.exp(-np.fabs(delta_t) / tau_plus)
        else:
            dw = a_minus(self.w) * np.exp(-np.fabs(delta_t) / tau_minus)
        return delta_t, dw

    def stdp_rule(self, t):
        t_pre, t_post = -1, -1
        if self.pre.spike_times:
            t_pre = self.pre.spike_times[-1]
        if self.post.spike_times:
            t_post = self.post.spike_times[-1]
        if t_post >= 0 and t_pre >= 0:
            delta_t, dw = self._stdp(t, t_pre, t_post)
            self.w += dw
            return delta_t, dw

    def rstdp_rule(self, t, dt):
        pass

    def update(self, learning_rule, t, dt):
        if learning_rule == "stdp":
            return self.stdp_rule(t)
        elif learning_rule == "rstdp":
            return self.rstdp_rule(t, dt)
        else:
            raise ValueError("INVALID LEARNING RULE")
