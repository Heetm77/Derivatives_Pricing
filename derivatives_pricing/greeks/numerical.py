"""
Numerical Greeks using finite differences.
"""

class NumericalGreeks:
    @staticmethod
    def delta(price_fn, S, h=1e-4):
        """
        Numerical Delta using central difference.

        Parameters
        ----------
        price_fn : callable
            Pricing function with signature price_fn(S)
        S : float
            Spot price
        h : float
            Small perturbation
        """
        return (price_fn(S + h) - price_fn(S - h)) / (2 * h)

    @staticmethod
    def gamma(price_fn, S, h=1e-4):
        """
        Numerical Gamma using central difference.
        """
        return (
            price_fn(S + h)
            - 2 * price_fn(S)
            + price_fn(S - h)
        ) / (h ** 2)

    @staticmethod
    def vega(price_fn, sigma, h=1e-4):
        """
        Numerical Vega using central difference.

        Parameters:
        price_fn : callable
            Pricing function with signature price_fn(sigma)
        sigma : float
            Volatility
        h : float
            Small perturbation
        """
        return (price_fn(sigma + h) - price_fn(sigma - h)) / (2 * h)

    @staticmethod
    def theta(price_fn, T, h=1e-5):
        """
        Numerical Theta using finite differences.

        Parameters
        price_fn : callable
            Pricing function with signature price_fn(T)
        T : float
            Time to maturity (years)
        h : float
            Small time step
        """
        if T <= h:
            # backward difference near expiry
            return (price_fn(T) - price_fn(T - h)) / h
        return (price_fn(T + h) - price_fn(T - h)) / (2 * h)
