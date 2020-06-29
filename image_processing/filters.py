import numpy as np


class DoG:
    def __init__(self, sigma1, sigma2):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def apply(self, size):
        radius = (size // 2, size // 2)
        [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
        g1 = (1 / self.sigma1) * np.exp(-(x ** 2 + y ** 2) / (2 * self.sigma1 ** 2))
        g2 = (1 / self.sigma2) * np.exp(-(x ** 2 + y ** 2) / (2 * self.sigma2 ** 2))
        dog = (1 / np.sqrt(2 * np.pi) * (g1 - g2))
        return dog


class Gabor:
    def __init__(self, lambd, theta, sigma, gamma):
        self.lambd = lambd
        self.theta = theta
        self.sigma = sigma
        self.gamma = gamma

    def apply(self, size):
        radius = (size // 2, size // 2)
        [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
        xx = x * np.cos(self.theta) + y * np.sin(self.theta)
        yy = -x * np.sin(self.theta) + y * np.cos(self.theta)
        gabor = np.exp(-(xx ** 2 + self.gamma ** 2 * yy ** 2) / (2 * self.sigma ** 2)) * \
            np.cos(2 * np.pi * xx / self.lambd)
        return gabor
