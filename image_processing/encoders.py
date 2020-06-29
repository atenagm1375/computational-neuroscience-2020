import numpy as np


class IntensityToLatency:
    def __init__(self, time_window, dt=1):
        self.time_window = time_window
        self.dt = dt

    def apply(self, image):
        time = self.time_window // self.dt
        shape = image.shape
        image /= np.max(image)

        time_to_spike = np.reciprocal(image, where=image != 0)
        time_to_spike *= (time / np.max(time_to_spike))
        return np.ceil(time_to_spike.reshape(shape))
