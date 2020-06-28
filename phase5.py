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

        plot_images([(image, "Original image"), (filtered_im, "DoG result")])


if __name__ == "__main__":
    q = sys.argv[1]
    t = sys.argv[2]

    globals()["q{}".format(q)](*globals()["q{}_t{}".format(q, t)]())
