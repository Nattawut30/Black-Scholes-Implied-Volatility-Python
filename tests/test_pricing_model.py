import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pricing_model import OptionsPricingModel, calculate_implied_volatility

def test_call_price_is_positive():
    model = OptionsPricingModel(100, 100, 1.0, 0.05, 0.20)
    assert model.call_price() > 0

def test_put_price_is_positive():
    model = OptionsPricingModel(100, 100, 1.0, 0.05, 0.20)
    assert model.put_price() > 0

def test_put_call_parity_holds():
    model = OptionsPricingModel(100, 100, 1.0, 0.05, 0.20)
    result = model.put_call_parity_check()
    assert result['is_valid'] == True

def test_implied_volatility_roundtrip():
    model = OptionsPricingModel(100, 100, 1.0, 0.05, 0.20)
    call = model.call_price()
    iv = calculate_implied_volatility(call, 100, 100, 1.0, 0.05, 'call')
    assert abs(iv - 0.20) < 0.001

def test_invalid_input_raises_error():
    try:
        OptionsPricingModel(-1, 100, 1.0, 0.05, 0.20)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
