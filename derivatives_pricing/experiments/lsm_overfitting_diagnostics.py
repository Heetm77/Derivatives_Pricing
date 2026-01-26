import numpy as np
import matplotlib.pyplot as plt

from derivatives_pricing.models.longstaff_schwartz import LongstaffSchwartzPricer


def run_overfitting_diagnostics():
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    n_paths = 50_000
    n_steps = 50
    seed = 42

    basis_degrees = [1, 2, 3, 4, 5]

    prices_naive = []
    prices_split = []

    for degree in basis_degrees:
        # Standard (in-sample) LSM
        price_naive = LongstaffSchwartzPricer.american_put(S0=S0, K=K, r=r, sigma=sigma, T=T,
            n_paths=n_paths, n_steps=n_steps, seed=seed, basis_degree=degree, split_paths=False)

        # Path-split (out-of-sample) LSM
        price_split = LongstaffSchwartzPricer.american_put(S0=S0, K=K, r=r, sigma=sigma, T=T,
            n_paths=n_paths, n_steps=n_steps, seed=seed, basis_degree=degree, split_paths=True)

        prices_naive.append(price_naive)
        prices_split.append(price_split)

        print(
            f"Basis degree {degree}: "
            f"Naive LSM = {price_naive:.4f}, "
            f"Path-split LSM = {price_split:.4f}"
        )

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(basis_degrees, prices_naive, marker='o', label='Naive LSM (in-sample)')
    plt.plot(basis_degrees, prices_split, marker='s', label='Path-split LSM (out-of-sample)')
    plt.xlabel("Regression Basis Degree")
    plt.ylabel("American Put Price")
    plt.title("LSM Overfitting Diagnostics via Path Splitting")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_overfitting_diagnostics()