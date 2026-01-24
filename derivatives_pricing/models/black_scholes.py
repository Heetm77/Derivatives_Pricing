import numpy as np
from scipy.stats import norm


class BlackScholes:
    """
    Black-Scholes option pricing model (European options).
    """

    @staticmethod
    def _d1(S, K, r, sigma, T):
        return (
            np.log(S / K)
            + (r + 0.5 * sigma ** 2) * T
        ) / (sigma * np.sqrt(T))

    @staticmethod
    def _d2(S, K, r, sigma, T):
        return BlackScholes._d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S, K, r, sigma, T):
        """
        Price a European call option using Black-Scholes.

        Parameters
        ----------
        S : float
            Spot price
        K : float
            Strike price
        r : float
            Risk-free rate (annualized)
        sigma : float
            Volatility (annualized)
        T : float
            Time to maturity (in years)
        """
        d1 = BlackScholes._d1(S, K, r, sigma, T)
        d2 = BlackScholes._d2(S, K, r, sigma, T)

        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, r, sigma, T):
        """
        European put option price using put-call parity.
        """
        call = BlackScholes.call_price(S, K, r, sigma, T)
        return call + K * np.exp(-r * T) - S
