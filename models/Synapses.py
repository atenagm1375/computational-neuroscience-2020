import numpy as np


class Synapse:
    def __init__(self, pre, post, weight, delay=1, trace_alpha=5, **parameters):
        self.pre = pre
        self.post = post
        self.w = weight
        self.d = delay
        self.parameters = parameters
        self.trace_alpha = trace_alpha

        self.a_plus = parameters.get("a_plus", lambda x: 0)
        self.a_minus = parameters.get("a_minus", lambda x: 0)
        self.tau_plus = parameters.get("tau_plus", 0)
        self.tau_minus = parameters.get("tau_minus", 0)

        self.c = parameters.get("c", 0)
        self.tau_c = parameters.get("tau_c", 0)
        self.tau_d = parameters.get("tau_d", 0)

    def _stdp(self, t, t_pre, t_post):
        delta_t = t_post - t_pre
        dw = 0
        if t == t_post:
            dw = self.a_plus(self.w) * \
                np.exp(-np.fabs(delta_t) / self.tau_plus)
        elif t == t_pre:
            dw = self.a_minus(self.w) * \
                np.exp(-np.fabs(delta_t) / self.tau_minus)
        return delta_t, dw

    def _rstdp(self, t, dt, t_pre, t_post, d, da):
        delta_t, stdp = self._stdp(t, t_pre, t_post)
        dc = (-self.c / self.tau_c + stdp) * dt
        dd = (-d / self.tau_d + da(t)) * dt
        self.c += dc
        d += dd
        dw = self.c * d * dt
        return delta_t, dw, d

    def update(self, learning_rule, t, dt, d=0, da=None):
        t_pre, t_post = -1, -1
        if self.pre.spike_times:
            t_pre = self.pre.spike_times[-1]
        if self.post.spike_times:
            t_post = self.post.spike_times[-1]
        if t_post >= 0 and t_pre >= 0:
            if learning_rule == "stdp":
                delta_t, dw = self._stdp(t, t_pre, t_post)
                self.w += dw
                return delta_t, dw
            elif learning_rule == "rstdp":
                delta_t, dw, d = self._rstdp(t, dt, t_pre, t_post, d, da)
                self.w += dw
                return delta_t, dw, d
            else:
                raise ValueError("INVALID LEARNING RULE")

    def __repr__(self):
        return self.w
