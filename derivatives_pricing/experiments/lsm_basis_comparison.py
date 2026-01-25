import numpy as np
import matplotlib.pyplot as plt
from derivatives_pricing.models.longstaff_schwartz import LongstaffSchwartzPricer


def run_basis_comparison():
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    n_paths = 50_000
    n_steps = 50
    seed = 42

    bases = {
        "Linear (1, S)": 1,
        "Quadratic (1, S, S²)": 2,
        "Cubic (1, S, S², S³)": 3,
    }

    prices = {}

    for name, degree in bases.items():
        price = LongstaffSchwartzPricer.american_put(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n_paths=n_paths,
            n_steps=n_steps,
            basis_degree=degree,
            seed=seed,
        )
        prices[name] = price
        print(f"{name}: {price:.4f}")

    # Plot comparison
    plt.figure()
    plt.bar(prices.keys(), prices.values())
    plt.ylabel("American Put Price")
    plt.title("LSM Price Sensitivity to Regression Basis")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_basis_comparison()