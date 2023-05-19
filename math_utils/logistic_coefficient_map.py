import numpy as np
# EXAMPLE 1:
# The map for the first N Taylor coefficients of the flow for ODE x' = x^2 - x
# This is a map of the form, T: R ---> R^N. In this simple case the map can be computed with any discrete convolution
# since the nonlinearity is only quadratic. Pretending this isn't the case, the best algorithm for computing T would
# require approximately N^2 floating point multiplications.


def logistic_taylor_orbit(x_0, N, tau=1.0):
    """Taylor coefficient map for the flow of the logistic with domain parameter tau computed via recursive formula."""

    def recursive_map(a):
        """The recursive Cauchy product terms for the logistic Taylor coefficient map"""
        return np.dot(a, np.flip(a)) - a[-1]

    taylorCoefficient = np.array([float(x_0)])  # initialize Taylor coefficient vector
    for j in range(N - 1):
        tNext = (tau / (j + 1)) * recursive_map(taylorCoefficient)
        taylorCoefficient = np.append(taylorCoefficient, tNext)

    return taylorCoefficient