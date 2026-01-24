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
