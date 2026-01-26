import numpy as np


class LongstaffSchwartzPricer:
    """
    American option pricing using Longstaff–Schwartz (LSM).
    Supports regression basis sensitivity and path-splitting diagnostics.
    """

    @staticmethod
    def american_put(
        S0, K, r, sigma, T,
        n_paths=100_000,
        n_steps=50,
        seed=None,
        basis_degree=2,
        split_paths=False,
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

        # Optional path splitting
        if split_paths:
            perm = np.random.permutation(n_paths)
            split = int(0.7 * n_paths)
            train_idx = perm[:split]
            test_idx = perm[split:]
        else:
            train_idx = test_idx = None

        # 3. Backward induction
        for t in range(n_steps - 1, 0, -1):
            itm = np.where(K - S[:, t] > 0)[0]
            otm = np.where(K - S[:, t] <= 0)[0]

            # OTM paths must continue
            V[otm, t] = discount * V[otm, t + 1]

            if len(itm) == 0:
                exercise_boundary.append(np.nan)
                continue

            # Select regression set
            if split_paths:
                itm_train = np.intersect1d(itm, train_idx)
                itm_test = np.intersect1d(itm, test_idx)

                if len(itm_train) == 0 or len(itm_test) == 0:
                    V[itm, t] = discount * V[itm, t + 1]
                    exercise_boundary.append(np.nan)
                    continue

                X_train = np.column_stack(
                    [S[itm_train, t] ** d for d in range(basis_degree + 1)]
                )
                Y_train = discount * V[itm_train, t + 1]

                beta = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]

                X_test = np.column_stack(
                    [S[itm_test, t] ** d for d in range(basis_degree + 1)]
                )
                continuation = X_test @ beta

                exercise = K - S[itm_test, t]
                exercise_now = exercise > continuation

                if np.any(exercise_now):
                    boundary = np.max(S[itm_test[exercise_now], t])
                else:
                    boundary = np.nan

                exercise_boundary.append(boundary)

                V[itm_test, t] = np.where(exercise_now, exercise, discount * V[itm_test, t + 1])
                exercised_paths = itm_test[exercise_now]
                V[exercised_paths, t + 1:] = 0.0

                # Train paths always continue
                V[itm_train, t] = discount * V[itm_train, t + 1]

            else:
                # Standard LSM (in-sample regression)
                X = np.column_stack(
                    [S[itm, t] ** d for d in range(basis_degree + 1)]
                )
                Y = discount * V[itm, t + 1]

                beta = np.linalg.lstsq(X, Y, rcond=None)[0]
                continuation = X @ beta

                exercise = K - S[itm, t]
                exercise_now = exercise > continuation

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
            return price, exercise_boundary[::-1]

        return price