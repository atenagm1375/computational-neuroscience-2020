from image_processing.filters import *


class Convolution:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def _conv2d(a, b):
        b = np.rot90(np.rot90(b))
        mul = a * b
        return np.sum(mul)

    def apply(self, image, kernel, stride=1, pad=True):
        if pad:
            new_image = np.zeros((image.shape[0] + stride, image.shape[1] + stride))
            new_image[stride:image.shape[0] - stride, stride:image.shape[1] - stride] = image
        else:
            new_image = image
        
        result = np.zeros(new_image.shape)
        for i in range(0, new_image.shape[0], stride):
            for j in range(0, new_image.shape[1], stride):
                subset = new_image[i:i + self.size, j:j + self.size]
                result[i:i + self.size, j:j + self.size] = self._conv2d(subset, kernel)

        return result
