import numpy as np



def schwefel(x):
    if x.ndim == 1:
        x = x[np.newaxis, :]
    n = x.shape[1]
    return 418.9829 * n - np.sum(
        x * np.sin(np.sqrt(np.abs(x))),
        axis=1)


def rosenbrock(x):
    if x.ndim == 1:
        x = x[np.newaxis, :]
    return np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 +
        (1.0 - x[:, :-1]) ** 2,
        axis=1)