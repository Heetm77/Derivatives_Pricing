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
