import numpy as np
from scipy import stats

def calculate_posterior_beta(prior_alpha, prior_beta, successes, trials):
    """Calculate posterior parameters for Beta-Binomial conjugate prior."""
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)
    return post_alpha, post_beta

def calculate_posterior_normal(prior_mean, prior_var, data_mean, data_var, n):
    """Calculate posterior parameters for Normal-Normal conjugate prior."""
    posterior_var = 1 / (1/prior_var + n/data_var)
    posterior_mean = posterior_var * (prior_mean/prior_var + n*data_mean/data_var)
    return posterior_mean, posterior_var