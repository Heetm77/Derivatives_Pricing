import numpy as np
from scipy.stats import norm
from derivatives_pricing.models.black_scholes import BlackScholes


class BlackScholesGreeks:
    """
    Analytical Greeks for Black-Scholes European options.
    """

    @staticmethod
    def delta_call(S, K, r, sigma, T):
        """
        Delta of a European call option.
        """
        d1 = BlackScholes._d1(S, K, r, sigma, T)
        return norm.cdf(d1)

    @staticmethod
    def delta_put(S, K, r, sigma, T):
        """
        Delta of a European put option.
        """
        d1 = BlackScholes._d1(S, K, r, sigma, T)
        return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S, K, r, sigma, T):
        """
        Gamma of a European option (same for call and put).
        """
        d1 = BlackScholes._d1(S, K, r, sigma, T)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
