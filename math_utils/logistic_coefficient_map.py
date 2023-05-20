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

# domain parameter definitions
def rescale(sequence, tau):
    """Generic function for rescaling a given parameterization sequence for an orbit of
    x' = f(x) into a parameterization of x' = tau*f(x)."""

    if tau == 1:
        return sequence
    else:
        powerVector = np.array([pow(tau, j) for j in range(len(sequence))])
        return sequence * powerVector


def tau_from_last_coef(sequence, mu=np.finfo(float).eps):
    """Set domain parameter by last coefficient decay. Input is a sequence computed with domain parameter equal to 1.
    Returns the domain parameter rescaling which forces the last coefficient norm to be equal to mu."""

    maxIdx = np.max(np.nonzero(sequence))  # index of the last nonzero coefficient
    tau = (mu / np.abs(sequence[maxIdx])) ** (1 / maxIdx)
    return tau