import pytest
import numpy as np
from src.pricing_model import BlackScholesModel

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

def test_option_strategy_payoff() -> None:
    
    stock_range = np.array([90.0, 100.0, 110.0])
    K = 100.0
    call_premium = 5.0
    put_premium = 4.0
    
    
    lc_payoff = BlackScholesModel.option_strategy_payoff("Long Call", stock_range, K, call_premium, put_premium)
    np.testing.assert_array_almost_equal(lc_payoff, [-5.0, -5.0, 5.0])


    straddle_payoff = BlackScholesModel.option_strategy_payoff("Long Straddle", stock_range, K, call_premium, put_premium)
    np.testing.assert_array_almost_equal(straddle_payoff, [1.0, -9.0, 1.0])
