import numpy as np
import matplotlib.pyplot as plt

from derivatives_pricing.models.black_scholes import BlackScholes
from derivatives_pricing.models.longstaff_schwartz import LongstaffSchwartzPricer


def early_exercise_experiment():
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    n_paths = 50_000
    n_steps = 50

    S_grid = np.linspace(60, 140, 15)

    american_prices = []
    european_prices = []

    for S0 in S_grid:
        american = LongstaffSchwartzPricer.american_put(
            S0, K, r, sigma, T,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=42
        )

        european = BlackScholes.put_price(
            S0, K, r, sigma, T
        )

        american_prices.append(american)
        european_prices.append(european)

    american_prices = np.array(american_prices)
    european_prices = np.array(european_prices)

    premium = american_prices - european_prices

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(S_grid, american_prices, label="American Put (LSM)", marker="o")
    plt.plot(S_grid, european_prices, label="European Put (BS)", marker="s")
    plt.plot(S_grid, premium, label="Early Exercise Premium", linestyle="--")

    plt.xlabel("Spot Price $S_0$")
    plt.ylabel("Option Value")
    plt.title("American vs European Put — Early Exercise Premium")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    early_exercise_experiment()