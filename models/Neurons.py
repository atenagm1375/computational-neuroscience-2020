import numpy as np


class LIF:
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50):
        self.tau = tau
        self.u_rest = u_rest
        self.r = r
        self.threshold = threshold

        self.__u = self.u_rest
        self.spike_times = []

    def simulate(self, current, t, dt):
        self.__u = __new_u(self, current, t, dt)
        if self.__u >= self.threshold:
            self.__u = self.u_rest
            self.spike_times.append(t)
        return self.__u

    def __new_u(self, current, t, dt):
        return self.__u + self.__tau_du_dt(current(t)) * self.tau * dt

    def __tau_du_dt(self, i):
        return -(self.__u - self.u_rest) + self.r * i


class ELIF(LIF):
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50, delta_t=1):
        super.__init__(tau, u_rest, r, threshold)
        self.delta_t = delta_t

    def __tau_du_dt(self, i):
        return super.__tau_du_dt(i) + self.delta_t * np.exp((self.__u - self.threshold) / self.delta_t)


class AddaptiveELIF(ELIF):
    def __init__(self, tau=10, u_rest=-70, r=5, threshold=-50, delta_t=1, tau_w=5, w=0, a=1, b=1):
        super.__init__(tau, u_rest, r, threshold, delta_t)
        self.tau_w = tau_w
        self.__w = w
        self.a = a
        self.b = b

    def __tau_du_dt(self, i):
        return super.__tau_du_dt(i) - self.r * self.__w

    def __tau_dw_dt(self, t):
        return self.a * (self.__u - self.u_rest) - self.__w + \
            self.b * self.tau_w * np.count_nonzero(self.spike_times == t)

    def simulate(self, current, t, dt):
        self.__u = __new_u(self, current, t, dt)
        self.__w = self.__w + self.__tau_dw_dt(t) * self.tau_w * dt
        if self.__u >= self.threshold:
            self.__u = self.u_rest
            self.spike_times.append(t)
        return self.__u
