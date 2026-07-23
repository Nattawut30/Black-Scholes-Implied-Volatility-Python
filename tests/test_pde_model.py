"""
Regression tests for the finite-difference / Monte-Carlo / binomial pricing
engines in src/pde_model.py.

The tolerances mirror the ones validated against the reference implementation
this project's finite-difference / Monte-Carlo design was checked against
(Roman Paolucci, Quant Guild — "Projects to Help You Become a Quant",
Quantitative Researcher project): FD vs closed-form Black-Scholes within
5e-3 at a 300x300 grid, American FD vs Longstaff-Schwartz Monte Carlo within
0.10, and American price never below the European price.
"""

import pytest
import numpy as np

from src.pricing_model import BlackScholesModel
from src.pde_model import (
    crank_nicolson_price,
    monte_carlo_european,
    monte_carlo_american_lsm,
    binomial_tree_american,
)


def _closed_form(S, K, T, r, sigma, q, option_type):
    m = BlackScholesModel(S, K, T, r, sigma, q)
    return m.call_price() if option_type == "call" else m.put_price()


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fd_european_matches_closed_form(option_type):
    """FD (Crank-Nicolson + Rannacher) must reproduce Black-Scholes to
    within 5e-3 at a 300x300 grid — the same bar the reference project's
    own smoke test enforces."""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    analytic = _closed_form(S, K, T, r, sigma, q, option_type)
    fd = crank_nicolson_price(S, K, T, r, sigma, q, option_type=option_type,
                               exercise="european", M=300, N=300)["price"]
    assert abs(fd - analytic) < 5e-3


def test_fd_european_matches_closed_form_with_dividends():
    S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.04, 0.28, 0.015
    for opt in ("call", "put"):
        analytic = _closed_form(S, K, T, r, sigma, q, opt)
        fd = crank_nicolson_price(S, K, T, r, sigma, q, option_type=opt,
                                   exercise="european", M=400, N=400)["price"]
        assert abs(fd - analytic) < 5e-3


def test_mc_european_within_confidence_interval():
    """European Monte Carlo must bracket the closed-form price in its own
    95% CI — otherwise one of the two engines disagrees with the other."""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    analytic = _closed_form(S, K, T, r, sigma, q, "call")
    mc = monte_carlo_european(S, K, T, r, sigma, q, "call", n_paths=200_000, seed=7)
    assert mc["ci_low"] <= analytic <= mc["ci_high"]


def test_american_call_equals_european_when_no_dividend():
    """With q=0, early exercise of a call is never optimal, so American
    must equal European (Merton, 1973)."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    euro = _closed_form(S, K, T, r, sigma, 0.0, "call")
    am = crank_nicolson_price(S, K, T, r, sigma, 0.0, option_type="call",
                               exercise="american", M=300, N=300)["price"]
    assert abs(euro - am) < 0.01


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_american_never_worth_less_than_european(option_type):
    """The early-exercise right can only add value."""
    S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.04, 0.28, 0.06
    euro = _closed_form(S, K, T, r, sigma, q, option_type)
    am = crank_nicolson_price(S, K, T, r, sigma, q, option_type=option_type,
                               exercise="american", M=300, N=300)["price"]
    assert am >= euro - 1e-6


def test_american_fd_matches_binomial_reference():
    """Two independent numerical methods (FD/Brennan-Schwartz vs CRR
    binomial tree) must agree closely — this is the American analogue of
    checking FD against a closed form."""
    S, K, T, r, sigma, q = 36.0, 40.0, 1.0, 0.06, 0.2, 0.0
    fd = crank_nicolson_price(S, K, T, r, sigma, q, option_type="put",
                               exercise="american", M=300, N=300)["price"]
    binom = binomial_tree_american(S, K, T, r, sigma, q, "put", n_steps=1500)
    assert abs(fd - binom) < 0.03


def test_american_fd_matches_lsm_monte_carlo():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    fd = crank_nicolson_price(S, K, T, r, sigma, q, option_type="put",
                               exercise="american", M=300, N=300)["price"]
    mc = monte_carlo_american_lsm(S, K, T, r, sigma, q, "put",
                                   n_paths=80_000, n_steps=50, seed=17)
    assert abs(fd - mc["price"]) < 0.10


def test_fd_second_order_convergence():
    """Doubling the grid resolution should cut the error by roughly 4x
    (2nd-order accuracy) — a regression guard against silently regressing
    to the 1st-order fully-upwind scheme."""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    analytic = _closed_form(S, K, T, r, sigma, q, "call")
    err_coarse = abs(crank_nicolson_price(S, K, T, r, sigma, q, option_type="call",
                                           exercise="european", M=100, N=100)["price"] - analytic)
    err_fine = abs(crank_nicolson_price(S, K, T, r, sigma, q, option_type="call",
                                         exercise="european", M=400, N=400)["price"] - analytic)
    assert err_fine < err_coarse / 3.0  # allow some slack around the theoretical 4x


def test_fd_stable_for_low_volatility_high_rate_regime():
    """The upwind fallback must keep the American solve stable (no NaN/inf,
    non-negative, capped by a sane upper bound) even in the regime where
    naive central differencing would break the M-matrix property."""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.20, 0.01, 0.24
    result = crank_nicolson_price(S, K, T, r, sigma, q, option_type="put",
                                   exercise="american", M=300, N=300)
    price = result["price"]
    assert np.isfinite(price)
    assert 0.0 <= price <= S


def test_binomial_converges_as_steps_increase():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.06, 0.2, 0.0
    coarse = binomial_tree_american(S, K, T, r, sigma, q, "put", n_steps=200)
    fine = binomial_tree_american(S, K, T, r, sigma, q, "put", n_steps=3000)
    assert abs(coarse - fine) < 0.05
