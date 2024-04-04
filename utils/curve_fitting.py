import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import curve_fit

@njit(nogil=True, parallel=True, cache=True)
def linear(x, a, b):
    return a + x*b

@njit(nogil=True, parallel=True, cache=True)
def linear_2(x, a, b):
    return (np.exp(a)*x**b)

@njit(nogil=True, parallel=True, cache=True)
def diminishing(x, a, b, c):
    return a + (x**b) / c

@njit(nogil=True, parallel=True, cache=True)
def diminishing_2(x, a, b):         #take log of both side to convert to linear function
    return np.exp(a)*(x**b)

@njit(nogil=True, parallel=True, cache=True)
def logarithm(x, a, b, c, d):
    return a + b*np.log(x*c) / np.log(d)

@njit(nogil=True, parallel=True, cache=True)
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

@njit(nogil=True, parallel=True, cache=True)
def sigmoid(x, A, k, x0, offset):
    return A/(1+np.exp(-k*(x-x0))) + offset

@njit(nogil=True, parallel=True, cache=True)
def polynomial(x, a, b, c):
    return a*x**2 + b*x + c

def bootstrap_by_index(X):
    return np.random.choice(np.arange(len(X)),
                            size=len(X),
                            replace=True
                            )

def bootstrap_fit(f, X, y, p0, trials):
    popts = []
    for i in np.arange(trials):
        shuffled_X = X.iloc[bootstrap_by_index(X)]
        shuffled_y = y.iloc[bootstrap_by_index(y)]
        popt, pcov = curve_fit(f,
                            shuffled_X,
                            shuffled_y,
                            p0
                            )
        popts.append(popt)
    return popts
    