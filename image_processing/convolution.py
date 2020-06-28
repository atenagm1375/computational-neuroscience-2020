from image_processing.filters import *


class Convolution:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def _conv2d(a, b):
        b = np.rot90(b, k=2)
        mul = a * b
        return np.sum(mul)

    def apply(self, image, kernel, stride=1, pad=True):
        if pad:
            padding = self.size // 2
            new_image = np.zeros((image.shape[0] + 2 * padding, image.shape[1] + 2 * padding))
            new_image[padding:image.shape[0] + padding, padding:image.shape[1] + padding] = image
        else:
            new_image = image
        
        result = np.zeros(new_image.shape)
        kernel = np.flipud(np.fliplr(kernel))
        for j in range(0, new_image.shape[1] - self.size + 1, stride):
            for i in range(0, new_image.shape[0] - self.size + 1, stride):
                subset = new_image[i:i + self.size, j:j + self.size]
                result[i, j] = self._conv2d(subset, kernel)

        return result
