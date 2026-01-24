import numpy as np
from derivatives_pricing.models.black_scholes import BlackScholes


def test_call_price_at_the_money():
    """
    ATM European call option.
    Reference value taken from standard Black-Scholes examples.
    """
    S = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    price = BlackScholes.call_price(S, K, r, sigma, T)

    # Known analytical value ≈ 10.4506
    assert np.isclose(price, 10.4506, atol=1e-4)


def test_call_price_deep_in_the_money():
    S = 150.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    price = BlackScholes.call_price(S, K, r, sigma, T)

    # Lower bound check: intrinsic value discounted
    intrinsic = S - K * np.exp(-r * T)
    assert price > intrinsic


def test_call_price_zero_volatility():
    """
    With zero volatility, option value should equal discounted intrinsic payoff.
    """
    S = 120.0
    K = 100.0
    r = 0.05
    sigma = 1e-8  # Avoid divide-by-zero
    T = 1.0

    price = BlackScholes.call_price(S, K, r, sigma, T)

    expected = max(S - K * np.exp(-r * T), 0)
    assert np.isclose(price, expected, atol=1e-4)


def test_put_price_at_the_money():
    S = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    price = BlackScholes.put_price(S, K, r, sigma, T)

    # Known analytical value ≈ 5.5735
    assert np.isclose(price, 5.5735, atol=1e-4)


def test_put_call_parity():
    S = 110.0
    K = 100.0
    r = 0.03
    sigma = 0.25
    T = 0.5

    call = BlackScholes.call_price(S, K, r, sigma, T)
    put = BlackScholes.put_price(S, K, r, sigma, T)

    lhs = call + K * np.exp(-r * T)
    rhs = put + S

    assert np.isclose(lhs, rhs, atol=1e-6)
