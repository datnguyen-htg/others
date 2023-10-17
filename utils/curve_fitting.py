import numpy as np
import pandas as pd
from numba import njit

@njit(nogil=True, parallel=True, cache=True)
def linear(x, a, b):
    return a + x*b

@njit(nogil=True, parallel=True, cache=True)
def diminishing(x, a, b, c):
    return a + b*x**c / c

@njit(nogil=True, parallel=True, cache=True)
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

@njit(nogil=True, parallel=True, cache=True)
def sigmoid(x, A, k, x0, offset):
    return A/(1+np.exp(-k*(x-x0))) + offset