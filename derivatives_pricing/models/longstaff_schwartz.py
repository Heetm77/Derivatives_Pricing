import numpy as np


class LongstaffSchwartzPricer:
    """
    American option pricing using Longstaff–Schwartz (LSM).
    """

    @staticmethod
    def american_put(
        S0, K, r, sigma, T,
        n_paths=100_000,
        n_steps=50,
        seed=None,
        basis_degree=2,
        return_boundary=False
    ):
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

        # 2. Initialize payoff
        V = np.zeros_like(S)
        V[:, -1] = np.maximum(K - S[:, -1], 0.0)

        exercise_boundary = []

        # 3. Backward induction
        for t in range(n_steps - 1, 0, -1):
            itm = np.where(K - S[:, t] > 0)[0]
            otm = np.where(K - S[:, t] <= 0)[0]

            # OTM paths: must continue
            V[otm, t] = discount * V[otm, t + 1]

            if len(itm) == 0:
                exercise_boundary.append(np.nan)
                continue

            # Regression basis
            X = np.column_stack(
                [S[itm, t] ** d for d in range(basis_degree + 1)]
            )

            Y = discount * V[itm, t + 1]

            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            continuation = X @ beta

            exercise = K - S[itm, t]
            exercise_now = exercise > continuation

            # Record boundary: max S where exercise is optimal
            if np.any(exercise_now):
                boundary = np.max(S[itm[exercise_now], t])
            else:
                boundary = np.nan

            exercise_boundary.append(boundary)

            V[itm, t] = np.where(exercise_now, exercise, Y)

            exercised_paths = itm[exercise_now]
            V[exercised_paths, t + 1:] = 0.0

        price = discount * np.mean(V[:, 1])

        if return_boundary:
            exercise_boundary = exercise_boundary[::-1]  # time order
            return price, exercise_boundary

        return price