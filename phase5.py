from tests.phase5.phase5 import *
import sys

from PIL import Image
from image_processing.filters import *
from image_processing.convolution import Convolution
from simulation.Monitors import plot_images


def q1(sigma1, sigma2, filter_sizes, k):
    image = np.asarray(Image.open('tests/phase5/car.jpg').resize((800, 400)).convert('L'))

    for n in filter_sizes:
        kernel = DoG(sigma1, sigma2).apply((n, n))
        conv = Convolution(n)
        filtered_im = conv.apply(image, kernel)

        plot_images(np.array([image, filtered_im]), np.array(["Original Image", "Image after DoG filter"]))


def q2(lambd, sigma, gamma, n_orientations, filter_sizes):
    image = np.asarray(Image.open('tests/phase5/car.jpg').resize((800, 400)).convert('L'))
    # plot_images(np.array([image]), np.array(["Original Image"]))

    filtered_images = []
    kernels = []
    titles = []
    for ind, n in enumerate(filter_sizes):
        filtered_images.append([])
        titles.append([])
        kernels.append([])
        for i in range(n_orientations):
            theta = i * np.pi / n_orientations
            kernel = Gabor(lambd, theta, sigma, gamma).apply((n, n))
            kernels[-1].append(kernel)
            conv = Convolution(n)
            filtered_im = conv.apply(image, kernel)
            filtered_images[-1].append(filtered_im)
            titles[-1].append("filter #{}: size={}, theta={}".format(ind, n, theta))

    plot_images(np.array(kernels))
    plot_images(np.array(filtered_images))


if __name__ == "__main__":
    q = sys.argv[1]
    t = sys.argv[2]

    globals()["q{}".format(q)](*globals()["q{}_t{}".format(q, t)]())
