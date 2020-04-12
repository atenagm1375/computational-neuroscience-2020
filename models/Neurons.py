import numpy as np


class LIF:
    def __init__(self, current=None, tau=10, u_rest=-70, r=5, threshold=-50):
        self.current = current
        self.tau = tau
        self.u_rest = u_rest
        self.r = r
        self.threshold = threshold

        self.__u = self.u_rest
        self.__t = 0
        self.__dt = 1

    def simulate(self, duration, time_step):
        pass

    def __new_u(self):
        pass
