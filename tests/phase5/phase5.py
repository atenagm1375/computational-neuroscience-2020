def q1_t1():
    sigma1 = 0.5
    sigma2 = 1.0
    sizes = [5, 13, 21, 29, 37]
    pad = False
    stride = 1
    k = 3
    return sigma1, sigma2, sizes, pad, stride, k


def q2_t1():
    lambd = 15
    sigma = 1
    gamma = 0.5
    n_orientations = 4
    filter_sizes = [5, 13, 21, 29, 37, 45, 53, 61]
    pad = False
    stride = 1
    time_window = 20
    return lambd, sigma, gamma, n_orientations, filter_sizes, pad, stride, time_window
