import numpy as np


class MonteCarloPricer:
    """
    Monte Carlo pricer for European options under GBM.
    """

    @staticmethod
    def simulate_terminal_price(S0, r, sigma, T, n_paths, seed=None):
        """
        Simulate terminal stock prices under GBM.
        """
        if seed is not None:
            np.random.seed(seed)

        Z = np.random.standard_normal(n_paths)
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * np.sqrt(T) * Z

        return S0 * np.exp(drift + diffusion)
