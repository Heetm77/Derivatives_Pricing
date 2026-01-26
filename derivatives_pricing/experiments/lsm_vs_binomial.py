import numpy as np
from derivatives_pricing.models.longstaff_schwartz import LongstaffSchwartzPricer
from derivatives_pricing.models.binomial import BinomialPricer


def run_benchmark():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

    lsm_price = LongstaffSchwartzPricer.american_put(S0, K, r, sigma, T, n_paths=100_000,
    n_steps=50, seed=42)

    binomial_steps = [50, 100, 200, 400]

    print("LSM American Put:", round(lsm_price, 4))
    print("\nBinomial Prices:")
    for n in binomial_steps:
        price = BinomialPricer.american_put(S0, K, r, sigma, T, n_steps=n)
        print(f"  Steps={n}: {price:.4f}")


if __name__ == "__main__":
    run_benchmark()