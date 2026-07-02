import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        
        assert model.put_call_parity_check() is True
