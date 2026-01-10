import numpy as np


# def schwefel(x):
#     x = np.asarray(x)
#     n = x.size
#     return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
#
#
# def rosenbrock(x):
#
#     x = np.asarray(x)
#     return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

# Zmiana funkcji na przyjmujace macierze
def schwefel(x):
    n = x.shape[1]
    return 418.9829 * n - np.sum(
        x * np.sin(np.sqrt(np.abs(x))),
        axis=1)


def rosenbrock(x):
    return np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 +
        (1.0 - x[:, :-1]) ** 2,
        axis=1)