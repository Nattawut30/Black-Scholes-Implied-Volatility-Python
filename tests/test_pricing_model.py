import pytest
from pricing_model import OptionsPricingModel, option_strategy_payoff
import numpy as np

def test_call_price_known_textbook_value():
    model = OptionsPricingModel(S=100, K=100, T=1, r=0.05, volatility=0.2)
    assert model.call_price() == pytest.approx(10.45, abs=0.01)

def test_put_call_parity_holds():
    model = OptionsPricingModel(S=100, K=100, T=1, r=0.05, volatility=0.2)
    assert model.put_call_parity_check()['is_valid'] is True

def test_strategy_payoff_put_differs_from_call():
    # This test locks in the exact bug you just fixed
    S_range = np.array([80, 100, 120])
    call_payoff = option_strategy_payoff(['long_call'], S_range, [100], [5], [1])
    put_payoff = option_strategy_payoff(['long_put'], S_range, [100], [5], [1])
    assert not np.array_equal(call_payoff, put_payoff)
