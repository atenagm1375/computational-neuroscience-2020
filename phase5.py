from tests.phase5.phase5 import *
import sys

from PIL import Image
from image_processing.filters import *
from image_processing.convolution import Convolution
from image_processing.encoders import IntensityToLatency
from simulation.Monitors import plot_images, plot_images_time_to_spike


def q1(sigma1, sigma2, filter_sizes, pad, stride, k):
    image = np.asarray(Image.open('tests/phase5/car.jpg').convert('L'))

    for n in filter_sizes:
        kernel = DoG(sigma1, sigma2).apply(n)
        conv = Convolution(n)
        filtered_im = conv.apply(image, kernel, pad=pad, stride=stride)

        plot_images(np.array([image, kernel, filtered_im]),
                    np.array(["Original Image", "kernel", "Image after DoG filter"]))


def q2(lambd, sigma, gamma, n_orientations, filter_sizes, pad, stride, time_window):
    image = np.asarray(Image.open('tests/phase5/car.jpg').convert('L'))

    filtered_images = []
    kernels = []
    titles = []
    time_to_spikes = []
    enc = IntensityToLatency(time_window)
    for ind, n in enumerate(filter_sizes):
        filtered_images.append([])
        titles.append([])
        kernels.append([])
        time_to_spikes.append([])
        for i in range(n_orientations):
            theta = i * np.pi / n_orientations
            kernel = Gabor(lambd, theta, sigma, gamma).apply(n)
            kernels[-1].append(kernel)
            conv = Convolution(n)
            filtered_im = conv.apply(image, kernel, pad=pad, stride=stride)
            filtered_images[-1].append(filtered_im)
            time_to_spikes[-1].append(enc.apply(filtered_im))
            titles[-1].append("filter #{}: size={}, theta={}".format(ind, n, theta))

    plot_images(np.array(kernels))
    plot_images(np.array(filtered_images))

    plot_images_time_to_spike(np.array(time_to_spikes))


if __name__ == "__main__":
    q = sys.argv[1]
    t = sys.argv[2]

    globals()["q{}".format(q)](*globals()["q{}_t{}".format(q, t)]())
