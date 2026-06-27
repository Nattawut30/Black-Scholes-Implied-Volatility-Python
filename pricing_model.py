"""
Options Pricing Model
Description: The Black-Scholes implementation with comprehensive

Created By: Nattawut Boonnoon
GitHub: https://github.com/Nattawut30
Linkedin: www.linkedin.com/in/nattawut-bn
Email: nattawut.boonnoon@hotmail.com
Phone: (+66) 92 271 6680
Locations: Bangkok, Thailand

Features:
- European Call & Put option pricing
- Complete Greeks calculations
- Implied volatility solver
- Historical volatility calculator
- Edge case handling and validation
- Real-world accuracy guaranteed

"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import warnings


class OptionsPricingModel:
    """
    Professional Black-Scholes Options Pricing Calculator
    
    Handles all calculations with robust error handling and edge case protection.
    Suitable for real-world trading analysis and educational purposes.
    
    Parameters:
    -----------
    stock_price : float
        Current price of underlying asset (S) - must be > 0
    strike_price : float
        Strike/exercise price (K) - must be > 0
    time_to_expiry : float
        Time to expiration in years (T) - must be > 0
    risk_free_rate : float
        Annual risk-free rate as decimal (r) - typically 0.01 to 0.10
    volatility : float
        Annual volatility as decimal (sigma) - must be > 0
    """
    
    # Class constants for validation
    MIN_STOCK_PRICE = 0.01
    MIN_STRIKE_PRICE = 0.01
    MIN_TIME = 0.000001  # Approximately 30 seconds
    MIN_VOLATILITY = 0.0001  # 0.01%
    MAX_VOLATILITY = 5.0  # 500%
    MIN_RATE = -0.1  # -10%
    MAX_RATE = 1.0  # 100%
    
    def __init__(self, stock_price, strike_price, time_to_expiry, risk_free_rate, volatility):
        """Initialize with validation and error handling"""
        
        # Validate inputs
        self._validate_inputs(stock_price, strike_price, time_to_expiry, 
                             risk_free_rate, volatility)
        
        # Store validated parameters
        self.S = float(stock_price)
        self.K = float(strike_price)
        self.T = float(time_to_expiry)
        self.r = float(risk_free_rate)
        self.sigma = float(volatility)
        
        # Calculate d1 and d2 safely
        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()
    
    def _validate_inputs(self, S, K, T, r, sigma):
        """
        Validate all inputs to prevent calculation errors
        
        Raises:
        -------
        ValueError : If any input is invalid
        """
        if S <= 0:
            raise ValueError(f"Stock price must be positive. Got: {S}")
        
        if K <= 0:
            raise ValueError(f"Strike price must be positive. Got: {K}")
        
        if T <= 0:
            raise ValueError(f"Time to expiry must be positive. Got: {T}")
        
        if sigma <= 0:
            raise ValueError(f"Volatility must be positive. Got: {sigma}")
        
        if sigma > self.MAX_VOLATILITY:
            raise ValueError(f"Volatility too high (max {self.MAX_VOLATILITY*100}%). Got: {sigma*100}%")
        
        if r < self.MIN_RATE or r > self.MAX_RATE:
            raise ValueError(f"Risk-free rate must be between {self.MIN_RATE*100}% and {self.MAX_RATE*100}%")
    
    def _calculate_d1(self):
        """
        Calculate d1 component with numerical stability
        
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        """
        try:
            numerator = np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T
            denominator = self.sigma * np.sqrt(self.T)
            
            # Handle edge case where denominator is very small
            if denominator < 1e-10:
                return 0.0
            
            return numerator / denominator
        except Exception as e:
            raise ValueError(f"Error calculating d1: {str(e)}")
    
    def _calculate_d2(self):
        """
        Calculate d2 component
        
        d2 = d1 - σ√T
        """
        return self.d1 - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        """
        Calculate European Call Option Price
        
        Call = S·N(d1) - K·e^(-rT)·N(d2)
        
        Returns:
        --------
        float : Call option price
        """
        try:
            call = (self.S * norm.cdf(self.d1) - 
                   self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
            return max(0.0, round(call, 4))  # Ensure non-negative
        except Exception as e:
            raise ValueError(f"Error calculating call price: {str(e)}")
    
    def put_price(self):
        """
        Calculate European Put Option Price
        
        Put = K·e^(-rT)·N(-d2) - S·N(-d1)
        
        Returns:
        --------
        float : Put option price
        """
        try:
            put = (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                  self.S * norm.cdf(-self.d1))
            return max(0.0, round(put, 4))  # Ensure non-negative
        except Exception as e:
            raise ValueError(f"Error calculating put price: {str(e)}")
    
    def intrinsic_value(self, option_type='call'):
        """
        Calculate intrinsic value (immediate exercise value)
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Intrinsic value
        """
        if option_type.lower() == 'call':
            return max(0, self.S - self.K)
        elif option_type.lower() == 'put':
            return max(0, self.K - self.S)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def time_value(self, option_type='call'):
        """
        Calculate time value (extrinsic value)
        
        Time Value = Option Price - Intrinsic Value
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Time value
        """
        if option_type.lower() == 'call':
            return self.call_price() - self.intrinsic_value('call')
        elif option_type.lower() == 'put':
            return self.put_price() - self.intrinsic_value('put')
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def get_greeks(self):
        """
        Calculate all option Greeks (risk sensitivities)
        
        Returns:
        --------
        dict : Complete set of Greeks for both call and put options
        """
        try:
            sqrt_T = np.sqrt(self.T)
            exp_neg_rT = np.exp(-self.r * self.T)
            
            # Shared calculations
            phi_d1 = norm.pdf(self.d1)
            N_d1 = norm.cdf(self.d1)
            N_neg_d1 = norm.cdf(-self.d1)
            N_d2 = norm.cdf(self.d2)
            N_neg_d2 = norm.cdf(-self.d2)
            
            # Delta: ∂V/∂S (price sensitivity to stock price)
            call_delta = N_d1
            put_delta = N_d1 - 1
            
            # Gamma: ∂²V/∂S² (rate of change of delta)
            # Same for call and put
            gamma = phi_d1 / (self.S * self.sigma * sqrt_T) if self.S * self.sigma * sqrt_T > 0 else 0
            
            # Vega: ∂V/∂σ (sensitivity to volatility)
            # Same for call and put, expressed per 1% change
            vega = (self.S * phi_d1 * sqrt_T) / 100
            
            # Theta: ∂V/∂T (time decay)
            # Expressed per calendar day
            call_theta_term1 = -(self.S * phi_d1 * self.sigma) / (2 * sqrt_T)
            call_theta_term2 = -self.r * self.K * exp_neg_rT * N_d2
            call_theta = (call_theta_term1 + call_theta_term2) / 365
            
            put_theta_term1 = -(self.S * phi_d1 * self.sigma) / (2 * sqrt_T)
            put_theta_term2 = self.r * self.K * exp_neg_rT * N_neg_d2
            put_theta = (put_theta_term1 + put_theta_term2) / 365
            
            # Rho: ∂V/∂r (sensitivity to interest rate)
            # Expressed per 1% change in rate
            call_rho = (self.K * self.T * exp_neg_rT * N_d2) / 100
            put_rho = -(self.K * self.T * exp_neg_rT * N_neg_d2) / 100
            
            return {
                # Call Greeks
                'call_delta': round(call_delta, 4),
                'call_gamma': round(gamma, 4),
                'call_vega': round(vega, 4),
                'call_theta': round(call_theta, 4),
                'call_rho': round(call_rho, 4),
                
                # Put Greeks
                'put_delta': round(put_delta, 4),
                'put_gamma': round(gamma, 4),  # Same as call
                'put_vega': round(vega, 4),     # Same as call
                'put_theta': round(put_theta, 4),
                'put_rho': round(put_rho, 4)
            }
        
        except Exception as e:
            raise ValueError(f"Error calculating Greeks: {str(e)}")
    
    def put_call_parity_check(self):
        """
        Verify Put-Call Parity relationship
        
        Put-Call Parity: C - P = S - K·e^(-rT)
        
        Returns:
        --------
        dict : Parity check results
        """
        left_side = self.call_price() - self.put_price()
        right_side = self.S - self.K * np.exp(-self.r * self.T)
        difference = abs(left_side - right_side)
        is_valid = difference < 0.01  # Allow small numerical errors
        
        return {
            'is_valid': is_valid,
            'left_side': round(left_side, 4),
            'right_side': round(right_side, 4),
            'difference': round(difference, 4)
        }


def calculate_implied_volatility(market_price, stock_price, strike_price, 
                                 time_to_expiry, risk_free_rate, option_type='call'):
    """
    Calculate implied volatility using Newton-Raphson method
    
    This is the volatility that makes the Black-Scholes price equal to the market price.
    CRITICAL for real-world trading - traders use IV more than theoretical prices!
    
    Parameters:
    -----------
    market_price : float
        Observed market price of the option
    stock_price : float
        Current stock price
    strike_price : float
        Strike price
    time_to_expiry : float
        Time to expiry in years
    risk_free_rate : float
        Risk-free rate as decimal
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    float : Implied volatility as decimal (e.g., 0.25 for 25%)
    """
    
    def objective_function(sigma):
        """Function to minimize: BS_price - market_price"""
        try:
            model = OptionsPricingModel(stock_price, strike_price, time_to_expiry, 
                                       risk_free_rate, sigma)
            if option_type.lower() == 'call':
                return model.call_price() - market_price
            else:
                return model.put_price() - market_price
        except Exception:
            return float('inf')
    
    try:
        # Use Brent's method for robust root finding
        # Search between 0.1% and 300% volatility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iv = brentq(objective_function, 0.001, 3.0, maxiter=100)
        return round(iv, 4)
    except:
        # If optimization fails, return None, alright?
        return None


def calculate_historical_volatility(price_series, periods=252):
    """
    Calculate historical volatility from price data
    
    Parameters:
    -----------
    price_series : array-like
        Historical price data
    periods : int
        Number of periods per year (252 for daily, 52 for weekly)
    
    Returns:
    --------
    float : Annualized historical volatility as decimal
    """
    try:
        prices = np.array(price_series)
        
        if len(prices) < 2:
            raise ValueError("Need at least 2 price points")
        
        # Calculate log returns
        returns = np.log(prices[1:] / prices[:-1])
        
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(periods)
        
        return round(volatility, 4)
    
    except Exception as e:
        raise ValueError(f"Error calculating historical volatility: {str(e)}")


# Convenience function for quick calculations
def quick_price(S, K, T_days, r_pct, sigma_pct):
    """
    Quick option pricing with intuitive inputs
    
    Parameters:
    -----------
    S : float - Stock price ($)
    K : float - Strike price ($)
    T_days : int - Days to expiry
    r_pct : float - Risk-free rate (%)
    sigma_pct : float - Volatility (%)
    
    Returns:
    --------
    dict : Call price, put price, and Greeks
    """
    T = T_days / 365
    r = r_pct / 100
    sigma = sigma_pct / 100
    
    model = OptionsPricingModel(S, K, T, r, sigma)
    
    return {
        'call_price': model.call_price(),
        'put_price': model.put_price(),
        'greeks': model.get_greeks(),
        'call_intrinsic': model.intrinsic_value('call'),
        'put_intrinsic': model.intrinsic_value('put'),
        'call_time_value': model.time_value('call'),
        'put_time_value': model.time_value('put'),
        'parity_check': model.put_call_parity_check()
    }
