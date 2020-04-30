import numpy as np


class LIF:
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50, is_inh=False, current=[]):
        self.tau = tau
        self.u_rest = u_rest
        self.r = r
        self.threshold = threshold
        self.current_list = current

        self._u = self.u_rest
        self.spike_times = []
        self.potential_list = []
        self.input = 0
        self.is_inh = is_inh

        self.target_synapses = []

    def _set_inh(self):
        self.is_inh = True
        return self

    def _current(self, t):
        return self.current_list[t]

    def step(self, t, dt):
        u = self.__new_u(self._current(int(t / dt)), dt)
        if u >= self.threshold:
            self.potential_list.append(self.threshold)
            u = self.u_rest
            self.spike_times.append(t)
            self.potential_list.append(u)
            for synapse in self.target_synapses:
                synapse.post.input += ((-1)**int(self.is_inh)
                                       * synapse.w * synapse.post._u)
        else:
            self.potential_list.append(u)
        self._u = u + self.input

    def __new_u(self, current, dt):
        return self._u + self._tau_du_dt(current) * (dt / self.tau)

    def _tau_du_dt(self, i):
        return -(self._u - self.u_rest) + self.r * i

    def input_reset(self, t, alpha):
        self.input -= alpha * self.input
        if self.input <= 0:
            self.input = 0


class ELIF(LIF):
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50, delta_t=1,
                 theta_rh=-55, is_inh=False, current=lambda x: x):
        super(ELIF, self).__init__(tau, u_rest, r, threshold, is_inh, current)
        self.delta_t = delta_t
        self.theta_rh = theta_rh

    def step(self, t, dt):
        u = self.__new_u(self._current(int(t // dt)), dt)
        if u >= self.threshold:
            self.potential_list.append(self.threshold)
            u = self.u_rest
            self.spike_times.append(t)
            self.potential_list.append(u)
            for synapse in self.target_synapses:
                synapse.post.input += ((-1)**int(self.is_inh) * synapse.w)
        else:
            self.potential_list.append(u)
        self._u = u + self.input

    def _tau_du_dt(self, i):
        return super(ELIF, self)._tau_du_dt(i) + self.delta_t * np.exp((self._u - self.theta_rh) / self.delta_t)

    def __new_u(self, current, dt):
        return self._u + self._tau_du_dt(current) * (dt / self.tau)


class AddaptiveELIF(ELIF):
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50, delta_t=1, theta_rh=-55,
                 tau_w=5, w=2, a=5, b=1, is_inh=False, current=lambda x: x):
        super(AddaptiveELIF, self).__init__(
            tau, u_rest, r, threshold, delta_t, theta_rh, is_inh, current)
        self.tau_w = tau_w
        self.__w = w
        self.a = a
        self.b = b

    def _tau_du_dt(self, i):
        return super(AddaptiveELIF, self)._tau_du_dt(i) - self.r * self.__w

    def _tau_dw_dt(self, t):
        return self.a * (self._u - self.u_rest) - self.__w + \
            self.b * self.tau_w * np.count_nonzero(self.spike_times == t)

    def step(self, t, dt):
        u = self.__new_u(self._current(int(t // dt)), dt)
        if u >= self.threshold:
            self.potential_list.append(self.threshold)
            u = self.u_rest
            self.spike_times.append(t)
            self.potential_list.append(u)
            for synapse in self.target_synapses:
                synapse.post.input += ((-1)**int(self.is_inh) * synapse.w)
        else:
            self.potential_list.append(u)
        self.__w = self.__w + self._tau_dw_dt(t) * (dt / self.tau_w)
        self._u = u + self.input

    def __new_u(self, current, dt):
        return self._u + self._tau_du_dt(current) * (dt / self.tau)
