import numpy as np


class LongstaffSchwartzPricer:
    """
    American option pricing using Longstaff–Schwartz (LSM).
    """

    @staticmethod
    def american_put(
        S0, K, r, sigma, T,
        n_paths=100_000,
        n_steps=50,
        seed=None
    ):
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        discount = np.exp(-r * dt)

        # Simulate GBM paths
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0

        Z = np.random.standard_normal((n_paths, n_steps))

        for t in range(1, n_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp(
                (r - 0.5 * sigma ** 2) * dt
                + sigma * np.sqrt(dt) * Z[:, t - 1]
            )

        # Terminal payoff (American put)
        V = np.maximum(K - S[:, -1], 0.0)

        return S, V
