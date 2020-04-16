import numpy as np


class LIF:
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50):
        self.tau = tau
        self.u_rest = u_rest
        self.r = r
        self.threshold = threshold

        self._u = self.u_rest
        self.spike_times = []

    def _simulate(self, current, t, dt):
        self._u = self.__new_u(current, t, dt)
        if self._u >= self.threshold:
            self._u = self.u_rest
            self.spike_times.append(t)
        return self._u

    def __new_u(self, current, t, dt):
        return self._u + self._tau_du_dt(current(t)) * (1 / self.tau) * dt

    def _tau_du_dt(self, i):
        return -(self._u - self.u_rest) + self.r * i


class ELIF(LIF):
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50, delta_t=1):
        super(ELIF, self).__init__(tau, u_rest, r, threshold)
        self.delta_t = delta_t

    def _simulate(self, current, t, dt):
        self._u = self.__new_u(current, t, dt)
        if self._u >= self.threshold:
            self._u = self.u_rest
            self.spike_times.append(t)
        return self._u

    def _tau_du_dt(self, i):
        return super(ELIF, self)._tau_du_dt(i) + self.delta_t * np.exp((self._u - self.threshold) / self.delta_t)

    def __new_u(self, current, t, dt):
        return self._u + self._tau_du_dt(current(t)) * (1 / self.tau) * dt


class AddaptiveELIF(ELIF):
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50, delta_t=1, tau_w=5, w=2, a=5, b=1):
        super(AddaptiveELIF, self).__init__(tau, u_rest, r, threshold, delta_t)
        self.tau_w = tau_w
        self.__w = w
        self.a = a
        self.b = b

    def _tau_du_dt(self, i):
        return super(AddaptiveELIF, self)._tau_du_dt(i) - self.r * self.__w

    def _tau_dw_dt(self, t):
        return self.a * (self._u - self.u_rest) - self.__w + \
            self.b * self.tau_w * np.count_nonzero(self.spike_times == t)

    def _simulate(self, current, t, dt):
        self._u = self.__new_u(current, t, dt)
        if self._u >= self.threshold:
            self._u = self.u_rest
            self.spike_times.append(t)
        self.__w = self.__w + self._tau_dw_dt(t) * (1 / self.tau_w) * dt
        return self._u

    def __new_u(self, current, t, dt):
        return self._u + self._tau_du_dt(current(t)) * (1 / self.tau) * dt
