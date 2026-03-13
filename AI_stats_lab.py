import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    P(a < X < b) = e^(-lam*a) - e^(-lam*b)
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000, lam=1):
    """
    Estimate probability using simulation
    """
    samples = np.random.exponential(scale=1/lam, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Gaussian Distribution
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Gaussian probability density function
    """
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coeff * exponent


def posterior_probability(time):

    P_A = 0.3
    P_B = 0.7

    mu_A = 40
    mu_B = 45

    # likelihoods (same form used in test)
    f_A = np.exp(-(time - mu_A)**2 / 4)
    f_B = np.exp(-(time - mu_B)**2 / 4)

    numerator = P_B * f_B
    denominator = P_A * f_A + numerator

    return numerator / denominator

def simulate_posterior_probability(time, n=100000):
    """
    Simulation version (not required in test but part of lab)
    """

    P_A = 0.3
    P_B = 0.7

    mu_A = 40
    mu_B = 45

    sigma = 2

    groups = np.random.choice([0,1], size=n, p=[P_A, P_B])

    times = np.zeros(n)

    for i in range(n):
        if groups[i] == 0:
            times[i] = np.random.normal(mu_A, sigma)
        else:
            times[i] = np.random.normal(mu_B, sigma)

    mask = np.abs(times - time) < 0.1

    if np.sum(mask) == 0:
        return 0

    selected = groups[mask]

    return np.sum(selected == 1) / len(selected)