import numpy as np
import matplotlib.pyplot as plt

from derivatives_pricing.models.black_scholes import BlackScholes
from derivatives_pricing.models.monte_carlo import MonteCarloPricer


def convergence_experiment():
    S = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    path_grid = np.logspace(3, 5, 15, dtype=int)

    mc_prices = []
    cv_prices = []

    for n in path_grid:
        mc_prices.append(
            MonteCarloPricer.european_call(
                S, K, r, sigma, T, n, seed=42
            )
        )
        cv_prices.append(
            MonteCarloPricer.european_call_control_variate(
                S, K, r, sigma, T, n, seed=42
            )
        )

    bs_price = BlackScholes.call_price(S, K, r, sigma, T)

    plt.figure(figsize=(10, 6))
    plt.plot(path_grid, mc_prices, marker='o', label='Monte Carlo')
    plt.plot(path_grid, cv_prices, marker='s', label='Control Variate')
    plt.axhline(bs_price, linestyle='--', label='Black-Scholes')

    plt.xscale('log')
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("Call Price")
    plt.title("Monte Carlo Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    convergence_experiment()
