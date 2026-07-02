"""
Options Pricing Model
Description: The Black-Scholes implementation with comprehensive

Created By: Nattawut Boonnoon
GitHub: https://github.com/Nattawut30
Linkedin: www.linkedin.com/in/nattawut-bn
Email: nattawut.boonnoon@hotmail.com
Locations: Bangkok, Thailand

Features:
- European Call & Put option pricing
- Complete Greeks calculations
- Implied volatility solver
- Historical volatility calculator
- Edge case handling and validation
- Built on standard Black-Scholes assumptions (see README for limitations)

Refactor: 02/07/2026

"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import warnings

class BlackScholesModel:
    """
    Professional Black-Scholes Options Pricing Calculator
    """
    MIN_STOCK_PRICE = 0.01
    MIN_STRIKE_PRICE = 0.01
    MIN_TIME = 0.000001  
    MIN_VOLATILITY = 0.0001  
    MAX_VOLATILITY = 5.0  
    MIN_RATE = -0.1  
    MAX_RATE = 1.0  

    def __init__(self, stock_price=100.0, strike_price=100.0, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.2, dividend_yield=0.0):
        """Initialize with validation and error handling"""
        self._validate_inputs(stock_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        
        self.S = float(stock_price)
        self.K = float(strike_price)
        self.T = float(time_to_expiry)
        self.r = float(risk_free_rate)
        self.sigma = float(volatility)
        self.q = float(dividend_yield)
        
        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()
    
    def _validate_inputs(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> None:
        if S < self.MIN_STOCK_PRICE:
            raise ValueError(f"Stock price (S) must be at least {self.MIN_STOCK_PRICE}")
        if K < self.MIN_STRIKE_PRICE:
            raise ValueError(f"Strike price (K) must be at least {self.MIN_STRIKE_PRICE}")
        if T < self.MIN_TIME:
            raise ValueError(f"Time to maturity (T) must be at least {self.MIN_TIME}")
        if sigma < self.MIN_VOLATILITY:
            raise ValueError(f"Volatility (sigma) must be at least {self.MIN_VOLATILITY}")
    
    def _calculate_d1(self):
        try:
            numerator = np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T
            denominator = self.sigma * np.sqrt(self.T)
            if denominator < 1e-10:
                return 0.0
            return numerator / denominator
        except Exception as e:
            raise ValueError(f"Error calculating d1: {str(e)}")
    
    def _calculate_d2(self):
        return self.d1 - self.sigma * np.sqrt(self.T)
    
    def call_price(self) -> float:
        try:
            call = (self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) -
                    self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
            return max(0.0, round(call, 4))
        except Exception as e:
            raise ValueError(f"Error calculating call price: {str(e)}")
    
    def put_price(self) -> float:
        try:
            put = (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) -
                   self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))
            return max(0.0, round(put, 4))
        except Exception as e:
            raise ValueError(f"Error calculating put price: {str(e)}")
    
    def intrinsic_value(self, option_type='call'):
        if option_type.lower() == 'call':
            return max(0, self.S - self.K)
        elif option_type.lower() == 'put':
            return max(0, self.K - self.S)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def time_value(self, option_type='call'):
        if option_type.lower() == 'call':
            return self.call_price() - self.intrinsic_value('call')
        elif option_type.lower() == 'put':
            return self.put_price() - self.intrinsic_value('put')
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def get_greeks(self):
        try:
            sqrt_T = np.sqrt(self.T)
            exp_neg_rT = np.exp(-self.r * self.T)
            phi_d1 = norm.pdf(self.d1)
            N_d1 = norm.cdf(self.d1)
            N_neg_d1 = norm.cdf(-self.d1)
            N_d2 = norm.cdf(self.d2)
            N_neg_d2 = norm.cdf(-self.d2)
            exp_neg_qT = np.exp(-self.q * self.T)

            call_delta = exp_neg_qT * N_d1
            put_delta = exp_neg_qT * (N_d1 - 1)
            gamma = (phi_d1 * exp_neg_qT) / (self.S * self.sigma * sqrt_T) if self.S * self.sigma * sqrt_T > 0 else 0
            vega = (self.S * exp_neg_qT * phi_d1 * sqrt_T) / 100

            call_theta = (
                -(self.S * exp_neg_qT * phi_d1 * self.sigma) / (2 * sqrt_T)
                - self.r * self.K * exp_neg_rT * N_d2
                + self.q * self.S * exp_neg_qT * N_d1
            ) / 365

            put_theta = (
                -(self.S * exp_neg_qT * phi_d1 * self.sigma) / (2 * sqrt_T)
                + self.r * self.K * exp_neg_rT * N_neg_d2
                - self.q * self.S * exp_neg_qT * N_neg_d1
            ) / 365

            call_rho = (self.K * self.T * exp_neg_rT * N_d2) / 100
            put_rho = -(self.K * self.T * exp_neg_rT * N_neg_d2) / 100
            vanna = (-exp_neg_qT * phi_d1 * (self.d2 / self.sigma)) if self.sigma > 0 else 0.0

            if sqrt_T > 0 and self.sigma > 0:
                charm = (-exp_neg_qT * phi_d1 * (
                    2 * (self.r - self.q) * self.T - self.d2 * self.sigma * sqrt_T
                ) / (2 * self.T * self.sigma * sqrt_T)) / 365
            else:
                charm = 0.0

            return {
                'call_delta': round(call_delta, 4),
                'call_gamma': round(gamma, 4),
                'call_vega': round(vega, 4),
                'call_theta': round(call_theta, 4),
                'call_rho': round(call_rho, 4),
                'put_delta': round(put_delta, 4),
                'put_gamma': round(gamma, 4),
                'put_vega': round(vega, 4),
                'put_theta': round(put_theta, 4),
                'put_rho': round(put_rho, 4),
                'vanna': round(vanna, 6),
                'charm': round(charm, 6)
            }
        except Exception as e:
            raise ValueError(f"Error calculating Greeks: {str(e)}")
    
    def put_call_parity_check(self):
        left_side = self.call_price() - self.put_price()
        right_side = self.S - self.K * np.exp(-self.r * self.T)
        difference = abs(left_side - right_side)
        is_valid = difference < 0.01
        
        return {
            'is_valid': is_valid,
            'left_side': round(left_side, 4),
            'right_side': round(right_side, 4),
            'difference': round(difference, 4)
        }

    def option_strategy_payoff(self, strategy_type: str, stock_price_range: np.ndarray, K: float, premium: float, spot_price: float | None = None) -> np.ndarray:
        strategy = strategy_type.lower().replace(" ", "_")
        if spot_price is None:
            spot_price = K 

        if strategy == 'long_call':
            return np.maximum(stock_price_range - K, 0) - premium
        elif strategy == 'long_put':
            return np.maximum(K - stock_price_range, 0) - premium
        elif strategy == 'covered_call':
            return (stock_price_range - spot_price) + (premium - np.maximum(stock_price_range - K, 0))
        elif strategy == 'protective_put':
            return (stock_price_range - spot_price) + (np.maximum(K - stock_price_range, 0) - premium)
        elif strategy == 'long_straddle':
            return np.maximum(stock_price_range - K, 0) + np.maximum(K - stock_price_range, 0) - premium
        elif strategy == 'short_straddle':
            return premium - np.maximum(stock_price_range - K, 0) - np.maximum(K - stock_price_range, 0)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

def calculate_implied_volatility(market_price: float, stock_price: float, strike_price: float, time_to_expiry: float, risk_free_rate: float, option_type: str = 'call', dividend_yield: float = 0.0) -> float | None:
    def objective_function(sigma):
        try:
            model = BlackScholesModel(stock_price, strike_price, time_to_expiry, risk_free_rate, sigma, dividend_yield=dividend_yield)
            if option_type.lower() == 'call':
                return model.call_price() - market_price
            else:
                return model.put_price() - market_price
        except Exception:
            return float('inf')
    try:
        return float(brentq(objective_function, 1e-6, 5.0))
    except (ValueError, RuntimeError) as e:
        import logging
        logging.error(f"Implied Volatility calculation failed due to mathematical limits: {e}")
        return None

def calculate_historical_volatility(price_series, periods=252):
    try:
        prices = np.array(price_series)
        if len(prices) < 2:
            raise ValueError("Need at least 2 price points")
        returns = np.log(prices[1:] / prices[:-1])
        volatility = np.std(returns) * np.sqrt(periods)
        return round(volatility, 4)
    except Exception as e:
        raise ValueError(f"Error calculating historical volatility: {str(e)}")

# Convenience function for quick calculations
def quick_price(S: float, K: float, T_days: int, r_pct: float, sigma_pct: float) -> dict:
    """
    Quick option pricing with intuitive inputs
    """
    T = T_days / 365
    r = r_pct / 100
    sigma = sigma_pct / 100


    model = BlackScholesModel(
        stock_price=S,
        strike_price=K,
        time_to_expiry=T,
        risk_free_rate=r,
        volatility=sigma
    )
    
    return {
        'call_price': model.call_price(),
        'put_price': model.put_price(),
        'greeks': model.get_greeks()
    }
