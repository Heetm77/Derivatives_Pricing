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

    @staticmethod
    def european_call(S0, K, r, sigma, T, n_paths, seed=None):
        ST = MonteCarloPricer.simulate_terminal_price(S0, r, sigma, T, n_paths, seed)
        payoff = np.maximum(ST - K, 0.0)
        return np.exp(-r * T) * np.mean(payoff)

    @staticmethod
    def european_put(S0, K, r, sigma, T, n_paths, seed=None):
        ST = MonteCarloPricer.simulate_terminal_price(S0, r, sigma, T, n_paths, seed)
        payoff = np.maximum(K - ST, 0.0)
        return np.exp(-r * T) * np.mean(payoff)

    @staticmethod
    def european_call_control_variate(S0, K, r, sigma, T, n_paths, seed=None):
        """
        Monte Carlo European call pricing using Black-Scholes as control variate.
        """
        if seed is not None:
            np.random.seed(seed)

        Z = np.random.standard_normal(n_paths)

        ST = S0 * np.exp(
            (r - 0.5 * sigma ** 2) * T
            + sigma * np.sqrt(T) * Z
        )

        payoff = np.exp(-r * T) * np.maximum(ST - K, 0.0)

        # Control variate: discounted ST
        control = np.exp(-r * T) * ST

        # Known expectation of control
        control_expectation = S0

        cov = np.cov(payoff, control, ddof=1)
        beta = cov[0, 1] / cov[1, 1]

        return np.mean(payoff + beta * (control_expectation - control))