import numpy as np
import matplotlib.pyplot as plt
from derivatives_pricing.models.longstaff_schwartz import LongstaffSchwartzPricer


def run_exercise_boundary():
    price, boundary = LongstaffSchwartzPricer.american_put(
        S0=100, K=100, r=0.05, sigma=0.2, T=1,
        n_paths=100_000,
        n_steps=50,
        seed=42,
        basis_degree=2,
        return_boundary=True
    )

    time = np.linspace(0, 1, len(boundary))

    plt.figure()
    plt.plot(time, boundary)
    plt.xlabel("Time to Maturity")
    plt.ylabel("Exercise Boundary S*")
    plt.title("LSM Estimated Optimal Exercise Boundary (American Put)")
    plt.grid()
    plt.show()

    print(f"American put price: {price:.4f}")


if __name__ == "__main__":
    run_exercise_boundary()