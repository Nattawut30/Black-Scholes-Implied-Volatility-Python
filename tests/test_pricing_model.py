import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pricing_model import BlackScholesModel

def test_textbook_reference_value() -> None:
    
    model = BlackScholesModel(stock_price=42.0, strike_price=40.0, time_to_expiry=0.5, risk_free_rate=0.1, volatility=0.2)
    call_price = model.black_scholes_call(S=42.0, K=40.0, T=0.5, r=0.1, sigma=0.2)
    assert pytest.approx(call_price, abs=1e-2) == 4.76

def test_put_call_parity_multiple_cases() -> None:
    
    model = BlackScholesModel(stock_price=42.0, strike_price=40.0, time_to_expiry=0.5, risk_free_rate=0.1, volatility=0.2)
    test_cases = [
        {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2},
        {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.1, "sigma": 0.2}
    ]
    for case in test_cases:
        call = model.black_scholes_call(case["S"], case["K"], case["T"], case["r"], case["sigma"])
        put = model.black_scholes_put(case["S"], case["K"], case["T"], case["r"], case["sigma"])
        assert model.put_call_parity_check(case["S"], case["K"], case["T"], case["r"], call, put) is True

def test_regression_long_put_payoff() -> None:
    
    model = BlackScholesModel(stock_price=42.0, strike_price=40.0, time_to_expiry=0.5, risk_free_rate=0.1, volatility=0.2)
    stock_prices = np.array([30.0, 40.0, 50.0])
    strike = 40.0
    premium = 2.0
    
    payoffs = model.option_strategy_payoff("long_put", stock_prices, strike, premium)
    
    
    assert payoffs[0] == 8.0
