from PIL import Image

from image_processing.filters import *
from image_processing.convolution import Convolution
from simulation.Monitors import plot_images


def q1(sigma1, sigma2, filter_sizes, k):
    image = np.asarray(Image.open('tests/phase5/car.jpg').convert('LA'))
    kernel = DoG(sigma1, sigma2)
    for n in filter_sizes:
        conv = Convolution(n)
        filtered_im = conv.apply(image, kernel)

        plot_images([(image, "Original image"), (filtered_im, "DoG result")])
