def q1_t1():
    sigma1 = 0.5
    sigma2 = 1.0
    sizes = [5, 13, 21, 29, 37]
    pad = False
    stride = 1
    k = 3
    return sigma1, sigma2, sizes, pad, stride, k


def q1_t2():
    sigma1 = 0.3
    sigma2 = 1.0
    sizes = [5, 13, 21, 29, 37]
    pad = False
    stride = 1
    k = 3
    return sigma1, sigma2, sizes, pad, stride, k


def q1_t3():
    sigma1 = 0.8
    sigma2 = 1.0
    sizes = [5, 13, 21, 29, 37]
    pad = False
    stride = 1
    k = 3
    return sigma1, sigma2, sizes, pad, stride, k


def q1_t4():
    sigma1 = 0.3
    sigma2 = 0.8
    sizes = [5, 13, 21, 29, 37]
    pad = False
    stride = 1
    k = 3
    return sigma1, sigma2, sizes, pad, stride, k


def q1_t5():
    sigma1 = 0.3
    sigma2 = 1.0
    sizes = [5, 13, 21, 29, 37]
    pad = False
    stride = 1
    k = 4
    return sigma1, sigma2, sizes, pad, stride, k


def q1_t6():
    sigma1 = 0.3
    sigma2 = 1.0
    sizes = [5, 13, 21, 29, 37]
    pad = False
    stride = 1
    k = 5
    return sigma1, sigma2, sizes, pad, stride, k


def q2_t1():
    lambd = 15
    sigma = 7.5
    gamma = 0.5
    n_orientations = 4
    filter_sizes = [7, 9, 11, 13, 15, 17, 19, 21, 23]
    pad = False
    stride = 1
    time_window = 15
    return lambd, sigma, gamma, n_orientations, filter_sizes, pad, stride, time_window


def q2_t2():
    lambd = 10
    sigma = 5
    gamma = 0.5
    n_orientations = 4
    filter_sizes = [7, 9, 11, 13, 15, 17, 19, 21, 23]
    pad = False
    stride = 1
    time_window = 15
    return lambd, sigma, gamma, n_orientations, filter_sizes, pad, stride, time_window


def q2_t3():
    lambd = 10
    sigma = 3
    gamma = 0.5
    n_orientations = 4
    filter_sizes = [7, 9, 11, 13, 15, 17, 19, 21, 23]
    pad = False
    stride = 1
    time_window = 15
    return lambd, sigma, gamma, n_orientations, filter_sizes, pad, stride, time_window


def q2_t4():
    lambd = 10
    sigma = 7
    gamma = 0.5
    n_orientations = 4
    filter_sizes = [7, 9, 11, 13, 15, 17, 19, 21, 23]
    pad = False
    stride = 1
    time_window = 15
    return lambd, sigma, gamma, n_orientations, filter_sizes, pad, stride, time_window


def q2_t5():
    lambd = 10
    sigma = 5
    gamma = 0.8
    n_orientations = 4
    filter_sizes = [7, 9, 11, 13, 15, 17, 19, 21, 23]
    pad = False
    stride = 1
    time_window = 15
    return lambd, sigma, gamma, n_orientations, filter_sizes, pad, stride, time_window


def q2_t6():
    lambd = 10
    sigma = 5
    gamma = 0.2
    n_orientations = 4
    filter_sizes = [7, 9, 11, 13, 15, 17, 19, 21, 23]
    pad = False
    stride = 1
    time_window = 15
    return lambd, sigma, gamma, n_orientations, filter_sizes, pad, stride, time_window
