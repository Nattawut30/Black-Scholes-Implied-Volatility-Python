import pytest
import numpy as np
from src.pricing_model import BlackScholesModel, calculate_implied_volatility

def test_textbook_reference_value() -> None:
    model = BlackScholesModel(
        stock_price=42.0,
        strike_price=40.0,
        time_to_expiry=0.5,
        risk_free_rate=0.1,
        volatility=0.2
    )
    call_price = model.call_price()
    assert pytest.approx(call_price, abs=1e-2) == 4.76

def test_put_call_parity_multiple_cases() -> None:
    test_cases = [
        {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2},
        {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.1, "sigma": 0.2}
    ]
    for case in test_cases:
        model = BlackScholesModel(
            stock_price=case["S"],
            strike_price=case["K"],
            time_to_expiry=case["T"],
            risk_free_rate=case["r"],
            volatility=case["sigma"]
        )
        call = model.call_price()
        put = model.put_price()
        
        left_side = call + case["K"] * np.exp(-case["r"] * case["T"])
        right_side = put + case["S"]
        assert pytest.approx(left_side, abs=1e-2) == right_side

def test_near_expiry_call_converges_to_intrinsic_value() -> None:
    """As T -> 0, the call price should converge to max(S-K, 0)."""
    model = BlackScholesModel(
        stock_price=110.0,
        strike_price=100.0,
        time_to_expiry=1e-5,
        risk_free_rate=0.05,
        volatility=0.2
    )
    assert pytest.approx(model.call_price(), abs=0.05) == 10.0


def test_near_expiry_out_of_the_money_call_is_near_zero() -> None:
    """An OTM option seconds before expiry should be worth ~0, not blow up."""
    model = BlackScholesModel(
        stock_price=90.0,
        strike_price=100.0,
        time_to_expiry=1e-5,
        risk_free_rate=0.05,
        volatility=0.2
    )
    assert model.call_price() == pytest.approx(0.0, abs=0.01)


def test_extreme_high_volatility_rejected() -> None:
    """sigma above MAX_VOLATILITY (5.0 / 500%) should raise, not silently run."""
    with pytest.raises(ValueError):
        BlackScholesModel(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=6.0
        )


def test_rate_outside_bounds_rejected() -> None:
    """Risk-free rate above MAX_RATE (1.0 / 100%) should raise, not silently run."""
    with pytest.raises(ValueError):
        BlackScholesModel(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=1.5,
            volatility=0.2
        )


def test_implied_volatility_returns_none_for_impossible_price() -> None:
    """A call can never be worth more than the stock price itself (S=100 here).
    A market price of 1000 is arbitrage-violating and has no solvable sigma,
    so the solver should return None, not raise or fabricate an answer."""
    iv = calculate_implied_volatility(
        market_price=1000.0,
        stock_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        option_type='call'
    )
    assert iv is None


def test_implied_volatility_round_trips_for_a_solvable_price() -> None:
    """Sanity check the solver's other branch: price an option at a known
    sigma, feed that price back in, and confirm we recover the same sigma."""
    known_sigma = 0.2
    priced = BlackScholesModel(100.0, 100.0, 1.0, 0.05, known_sigma).call_price()
    iv = calculate_implied_volatility(
        market_price=priced,
        stock_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        option_type='call'
    )
    assert iv == pytest.approx(known_sigma, abs=1e-3)
def test_put_call_parity_with_dividend_yield() -> None:
    """Regression test: put_call_parity_check() must discount the spot side
    by the dividend yield (S * e^-qT), not just use raw S. Without this,
    the parity check falsely flags correctly-priced options as invalid
    whenever dividend_yield != 0."""
    model = BlackScholesModel(
        stock_price=100.0,
        strike_price=105.0,
        time_to_expiry=0.75,
        risk_free_rate=0.04,
        volatility=0.28,
        dividend_yield=0.015
    )
    result = model.put_call_parity_check()
    assert bool(result["is_valid"]) is True
    assert result["difference"] < 0.01

def test_scenario_analysis_shape_and_monotonicity():
    from src.pricing_model import scenario_analysis
    rows = scenario_analysis(S=100, K=100, T_days=180, r_pct=5, sigma_pct=25, q_pct=0)
    assert len(rows) == 7
    call_prices = [r['call_price'] for r in rows]
    put_prices = [r['put_price'] for r in rows]
    assert all(call_prices[i] <= call_prices[i + 1] for i in range(len(call_prices) - 1))
    assert all(put_prices[i] >= put_prices[i + 1] for i in range(len(put_prices) - 1))

def test_scenario_analysis_extreme_negative_shock_does_not_crash():
    from src.pricing_model import scenario_analysis
    rows = scenario_analysis(S=0.02, K=100, T_days=180, r_pct=5, sigma_pct=25, shocks=(-0.99,))
    assert rows[0]['stock_price'] >= 0.01

@pytest.mark.parametrize(
    "sigma",
    [0.0, -0.2, BlackScholesModel.MIN_VOLATILITY - 1e-8],
)
def test_volatility_below_minimum_is_rejected(sigma: float) -> None:
    with pytest.raises(ValueError, match="Volatility"):
        BlackScholesModel(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=sigma,
        )


def test_minimum_volatility_is_accepted() -> None:
    model = BlackScholesModel(
        stock_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=BlackScholesModel.MIN_VOLATILITY,
    )
    assert model.call_price() >= 0.0
