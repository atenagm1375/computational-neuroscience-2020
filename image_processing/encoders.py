import numpy as np


class IntensityToLatency:
    def __init__(self, time_window, dt=1):
        self.time_window = time_window
        self.dt = dt

    def apply(self, image):
        shape = image.shape
        # image = np.fabs(image)
        image = image * (image > 0)
        interval = np.max(image) - np.min(image)
        # image = image - np.min(image)
        time_to_spike = np.zeros(shape)
        time = self.time_window // self.dt
        interval /= time
        for t in range(time):
            im1 = image >= t * interval
            im2 = image < (t + 1) * interval
            im = im1 * im2
            time_to_spike += (time - t) * im
        return time_to_spike


class KSplit:
    def __init__(self, k):
        self.k = k

    def apply(self, image):
        a = image * (image > 0)
        a -= np.mean(a)
        interval = np.max(a) - np.min(a)
        res = []
        for i in range(self.k):
            im1 = i * (interval / self.k) < a
            im2 = a < (i + 1) * (interval / self.k)
            im = im1 * im2
            res.append(im)
        return res
