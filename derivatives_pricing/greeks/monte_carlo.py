import numpy as np
from derivatives_pricing.models.monte_carlo import MonteCarloPricer


class MonteCarloGreeks:
    """
    Monte Carlo estimators for option Greeks.
    """

    @staticmethod
    def delta_bump_and_revalue(
        S0, K, r, sigma, T, n_paths, eps=1e-4, seed=None
    ):
        """
        Delta via bump-and-revalue using central differences.
        """
        price_up = MonteCarloPricer.european_call(
            S0 + eps, K, r, sigma, T, n_paths, seed
        )
        price_down = MonteCarloPricer.european_call(
            S0 - eps, K, r, sigma, T, n_paths, seed
        )

        return (price_up - price_down) / (2 * eps)

    @staticmethod
    def delta_pathwise(
        S0, K, r, sigma, T, n_paths, seed=None
    ):
        pass
