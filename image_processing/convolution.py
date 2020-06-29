from image_processing.filters import *


class Convolution:
    def __init__(self, size):
        self.size = int(size)

    def apply(self, image, kernel, stride=1, pad=True):
        if pad:
            padding = self.size // 2
            new_image = np.zeros((image.shape[0] + 2 * padding, image.shape[1] + 2 * padding))
            new_image[padding:-padding, padding:-padding] = image
        else:
            padding = 0
            new_image = np.zeros(image.shape)
            new_image[:, :] = image

        outx = (image.shape[0] - self.size + 2 * padding) // stride + 1
        outy = (image.shape[1] - self.size + 2 * padding) // stride + 1
        result = np.zeros((outx, outy))
        kernel = np.flipud(np.fliplr(kernel))
        for i in range(new_image.shape[0]):
            if i > new_image.shape[0] - self.size:
                break
            if i % stride == 0:
                for j in range(new_image.shape[1]):
                    if j > new_image.shape[1] - self.size:
                        break
                    subset = new_image[i:i + self.size, j:j + self.size]
                    if i % stride == 0:
                        result[i, j] = np.sum(kernel * subset)
                    else:
                        break

        return result
