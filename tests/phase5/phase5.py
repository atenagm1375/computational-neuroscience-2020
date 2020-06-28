def q1_t1():
    sigma1 = 0.3
    sigma2 = 1.0
    sizes = [3, 7, 11, 15, 19]
    k = 3
    return sigma1, sigma2, sizes, k


def q2_t1():
    lambd = 0.2
    sigma = 0.1
    gamma = 0.01
    n_orientations = 4
    filter_sizes = [3, 5, 7, 9, 11, 13, 15, 17]
    time_window = 20
    return lambd, sigma, gamma, n_orientations, filter_sizes, time_window
