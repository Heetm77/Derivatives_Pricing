import numpy as np


class BinomialPricer:
    """
    Binomial tree pricing for American options (CRR).
    """

    @staticmethod
    def american_put(S0, K, r, sigma, T, n_steps=200):
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)

        # Terminal prices
        S = S0 * d ** np.arange(n_steps, -1, -1) * u ** np.arange(0, n_steps + 1)
        V = np.maximum(K - S, 0.0)

        # Backward induction
        for _ in range(n_steps):
            V = discount * (p * V[1:] + (1 - p) * V[:-1])
            S = S[:-1] / d
            V = np.maximum(V, K - S)

        return V[0]