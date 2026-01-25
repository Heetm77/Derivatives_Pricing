import numpy as np


class LongstaffSchwartzPricer:
    """
    American option pricing using Longstaff–Schwartz (LSM).
    """

    @staticmethod
    def american_put(
        S0,
        K,
        r,
        sigma,
        T,
        n_paths=100_000,
        n_steps=50,
        basis_degree=2,
        seed=None,
    ):
        if basis_degree < 1:
            raise ValueError("basis_degree must be >= 1")

        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        discount = np.exp(-r * dt)

        # 1. Simulate GBM paths
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0

        Z = np.random.standard_normal((n_paths, n_steps))

        for t in range(1, n_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp(
                (r - 0.5 * sigma ** 2) * dt
                + sigma * np.sqrt(dt) * Z[:, t - 1]
            )

        # 2. Initialize payoff matrix
        V = np.zeros_like(S)
        V[:, -1] = np.maximum(K - S[:, -1], 0.0)

        # 3. Backward induction
        for t in range(n_steps - 1, 0, -1):
            itm = np.where(K - S[:, t] > 0)[0]
            otm = np.where(K - S[:, t] <= 0)[0]

            # OTM paths: must continue
            V[otm, t] = discount * V[otm, t + 1]

            if len(itm) == 0:
                continue

            # Regression basis: [1, S, S^2, ..., S^basis_degree]
            X = np.column_stack(
                [S[itm, t] ** d for d in range(basis_degree + 1)]
            )

            Y = discount * V[itm, t + 1]

            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            continuation = X @ beta

            exercise_value = K - S[itm, t]
            exercise_now = exercise_value > continuation

            # Decide whether to exercise or continue
            V[itm, t] = np.where(exercise_now, exercise_value, Y)

            # Zero out future cashflows for exercised paths
            exercised_paths = itm[exercise_now]
            V[exercised_paths, t + 1:] = 0.0

        # 4. Discount to time 0
        price = discount * np.mean(V[:, 1])
        return price