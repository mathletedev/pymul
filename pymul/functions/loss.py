import numpy as np


def mse(targets, outputs):
    return np.mean(np.power(targets - outputs, 2))


def mse_prime(targets, outputs):
    return 2 * (outputs - targets) / targets.size
