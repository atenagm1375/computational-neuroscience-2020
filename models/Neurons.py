import numpy as np

import heapq


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
        self.time_to_apply = []
        heapq.heapify(self.time_to_apply)

    def _set_inh(self):
        self.is_inh = True
        return self

    def _current(self, t):
        return self.current_list[t]

    def apply_pre_synaptic(self, t, dt):
        du = 0
        if len(self.time_to_apply) > 0 and self.time_to_apply[0] == t:
            heapq.heappop(self.time_to_apply)
            # self.time_to_apply.pop(0)
            du += self.input
            self.potential_list[-1] += du
        return du

    def _check_spike(self, t, dt):
        u = self.potential_list[-1]
        if u >= self.threshold:
            self.potential_list.pop(-1)
            self.potential_list.append(self.threshold)
            u = self.u_rest
            self.spike_times.append(t)
            self.potential_list.append(u)
            for synapse in self.target_synapses:
                # synapse.post.time_to_apply.append(t + synapse.d)
                if t + synapse.d * dt not in self.time_to_apply:
                    heapq.heappush(synapse.post.time_to_apply,
                                   t + synapse.d * dt)
                synapse.post.input += (-1)**int(self.is_inh) * \
                    synapse.w * (self.threshold - self.u_rest)
        return u

    def compute_spike(self, t, dt):
        self._u = self._check_spike(t, dt)

    def compute_potential(self, t, dt):
        u = self.__new_u(self._current(int(t / dt)), dt)
        self.potential_list.append(u)
        return u

    # def step(self, t, dt):
    #     u = self.__new_u(self._current(int(t / dt)), dt)
    #     # u += _apply_pre_synaptic(t)
    #     if u >= self.threshold:
    #         self.potential_list.append(self.threshold)
    #         u = self.u_rest
    #         self.spike_times.append(t)
    #         self.potential_list.append(u)
    #         for synapse in self.target_synapses:
    #             synapse.post.input += ((-1)**int(self.is_inh)
    #                                    * synapse.w * (self.threshold - self.u_rest))
    #     else:
    #         self.potential_list.append(u)
    #     self._u = u

    def __du(self, current, dt):
        return self._tau_du_dt(current) * (dt / self.tau)

    def __new_u(self, current, dt):
        return self._u + self.__du(current, dt)

    def _tau_du_dt(self, i):
        return -(self._u - self.u_rest) + self.r * i

    def input_reset(self, t, dt, alpha):
        # self._u += self.input]
        self.input -= alpha * self.input
        if self.input <= 0:
            self.input = 0


class ELIF(LIF):
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50, delta_t=1,
                 theta_rh=-55, is_inh=False, current=lambda x: x):
        super(ELIF, self).__init__(tau, u_rest, r, threshold, is_inh, current)
        self.delta_t = delta_t
        self.theta_rh = theta_rh

    def _apply_pre_synaptic(self, t, dt):
        du = 0
        if self.time_to_apply[0] == t:
            self.time_to_apply.pop(0)
            du += self.__du(self.input, dt)
        return du

    def compute_potential(self, t, dt):
        u = self.__new_u(self._current(int(t / dt)), dt)
        self.potential_list.append(u)
        return u

    # def step(self, t, dt):
    #     u = self.__new_u(self._current(int(t // dt)), dt)
    #     # u += _apply_pre_synaptic(t)
    #     if u >= self.threshold:
    #         self.potential_list.append(self.threshold)
    #         u = self.u_rest
    #         self.spike_times.append(t)
    #         self.potential_list.append(u)
    #         for synapse in self.target_synapses:
    #             synapse.post.input += (-1)**int(self.is_inh) * synapse.w
    #     else:
    #         self.potential_list.append(u)
    #     self._u = u

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

    def _apply_pre_synaptic(self, t, dt):
        du = 0
        if self.time_to_apply[0] == t:
            self.time_to_apply.pop(0)
            du += self.__du(self.input, dt)
        return du

    def compute_potential(self, t, dt):
        u = self.__new_u(self._current(int(t / dt)), dt)
        self.potential_list.append(u)
        return u

    def compute_spike(self, t, dt):
        u, du = self._check_spike(t, dt)
        self.__w = self.__w + self._tau_dw_dt(t) * (dt / self.tau_w)
        self._u = u + du

    def _tau_du_dt(self, i):
        return super(AddaptiveELIF, self)._tau_du_dt(i) - self.r * self.__w

    def _tau_dw_dt(self, t):
        return self.a * (self._u - self.u_rest) - self.__w + \
            self.b * self.tau_w * np.count_nonzero(self.spike_times == t)

    # def step(self, t, dt):
    #     u = self.__new_u(self._current(int(t // dt)), dt)
    #     # u += _apply_pre_synaptic(t)
    #     if u >= self.threshold:
    #         self.potential_list.append(self.threshold)
    #         u = self.u_rest
    #         self.spike_times.append(t)
    #         self.potential_list.append(u)
    #         for synapse in self.target_synapses:
    #             synapse.post.input += (-1)**int(self.is_inh) * synapse.w
    #     else:
    #         self.potential_list.append(u)
    #     self.__w = self.__w + self._tau_dw_dt(t) * (dt / self.tau_w)
    #     self._u = u

    def __new_u(self, current, dt):
        return self._u + self._tau_du_dt(current) * (dt / self.tau)
