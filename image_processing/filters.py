import numpy as np


class DoG:
    def __init__(self, sigma1, sigma2):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def apply(self, size):
        dog = np.zeros(size)
        for x in range(size[0]):
            for y in range(size[1]):
                g1 = (1 / self.sigma1) * np.exp(-(x ** 2 + y ** 2) / (2 * self.sigma1 ** 2))
                g2 = (1 / self.sigma2) * np.exp(-(x ** 2 + y ** 2) / (2 * self.sigma2 ** 2))
                dog[x][y] = (1 / np.sqrt(2 * np.pi) * (g1 - g2))
        return dog


class Gabor:
    def __init__(self, lambd, theta, sigma, gamma):
        self.lambd = lambd
        self.theta = theta
        self.sigma = sigma
        self.gamma = gamma

    def apply(self, size):
        gabor = np.zeros(size)
        for x in range(size[0]):
            for y in range(size[1]):
                xx = x * np.cos(self.theta) + y * np.sin(self.theta)
                yy = -x * np.sin(self.theta) + y * np.cos(self.theta)
                gabor[x][y] = np.exp(-(xx ** 2 + self.gamma ** 2 * yy ** 2) / (2 * self.sigma ** 2)) * np.cos(
                    2 * np.pi * xx / self.lambd)
        return gabor
