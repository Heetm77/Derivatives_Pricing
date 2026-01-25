import numpy as np
import matplotlib.pyplot as plt

from derivatives_pricing.models.longstaff_schwartz import LongstaffSchwartzPricer


def run_convergence():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1

    paths_list = [5_000, 10_000, 20_000, 50_000, 100_000]
    prices = []

    for n in paths_list:
        price = LongstaffSchwartzPricer.american_put(
            S0, K, r, sigma, T,
            n_paths=n,
            n_steps=50,
            seed=42
        )
        prices.append(price)

    plt.plot(paths_list, prices, marker="o")
    plt.xlabel("Number of paths")
    plt.ylabel("American put price")
    plt.title("LSM Convergence (Paths)")
    plt.show()


if __name__ == "__main__":
    run_convergence()