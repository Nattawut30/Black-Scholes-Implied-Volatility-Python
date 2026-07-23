"""
Black-Scholes PDE Solver — numerical pricing engines
=====================================================

Three independent ways to price an option, cross-checked against each other
in the UI (by the Feynman-Kac theorem, the PDE and the discounted risk-neutral
expectation are the SAME quantity — if both engines are implemented correctly
they must agree to within numerical/sampling error):

1. Finite Difference (theta-scheme) — solves the Black-Scholes PDE directly:

        dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0

   European: Crank-Nicolson (theta=1/2), 2nd-order accurate in both S and
   time, with two Rannacher start-up steps (Rannacher, 1984) run fully
   implicit to damp the spurious oscillations Crank-Nicolson otherwise
   produces near a non-smooth payoff (the kink at the strike).

   American: fully implicit (theta=1) combined with the Brennan-Schwartz
   (1977) projected tridiagonal solve, which enforces the early-exercise
   constraint V(S,t) >= payoff(S) exactly at every node. Fully implicit
   (rather than Crank-Nicolson) is used deliberately for American: CN does
   not damp the free-boundary oscillations as well, a well-known FD-for-
   American pitfall.

2. Monte Carlo:
     - European: exact terminal-distribution simulation with antithetic
       variates (no discretization bias — only S_T matters for the payoff).
     - American: Longstaff-Schwartz Least-Squares Monte Carlo (LSM),
       Longstaff & Schwartz (2001).

3. Reference / ground truth:
     - European: closed-form Black-Scholes (BlackScholesModel, untouched).
     - American call, q=0: equals the European call (never optimal to
       exercise early with no dividends) — so the closed-form call IS the
       reference, no numerics needed.
     - American put (or American call with q>0): no closed form exists.
       A Cox-Ross-Rubinstein binomial tree at high step count is used as an
       independent third benchmark, on top of FD and MC agreeing with
       each other.

Numerical note on the finite-difference scheme
-----------------------------------------------
Plain central differencing of the advection term (r-q)*S*dV/dS can produce
a NEGATIVE off-diagonal coefficient near S=0 whenever the drift is large
relative to sigma^2 there — breaking the M-matrix property early-exercise
projection (and monotonicity generally) relies on. This is a known failure
mode, and it is not hypothetical here: this app's sliders allow rates up to
20% with volatility as low as 1%, which triggers it. The fix (Duffy,
"Finite Difference Methods in Financial Engineering") is to fall back to
upwind differencing (1st-order, but unconditionally sign-safe) only at the
specific nodes where central differencing would go negative, and use
central differencing (2nd-order) everywhere else. See _fd_coefficients().
"""

import numpy as np

# Reuse the exact same input bounds as the closed-form model so the two
# engines never silently disagree about what counts as a "valid" input.
# Try the bare import first (how streamlit_app.py imports it — Streamlit
# runs with src/ as the working directory), then fall back to the
# package-qualified import (how pytest imports it via `from src.pde_model
# import ...`).
try:
    from pricing_model import BlackScholesModel
except ImportError:
    from src.pricing_model import BlackScholesModel

_Z95 = 1.959963984540054  # exact two-sided 95% normal quantile


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _payoff(S, K, option_type):
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def _fd_coefficients(i, sigma, r, q):
    """
    Coefficients (p_i, q_i, s_i) of the spatial operator

        L[V]_i = p_i * V_{i-1} + q_i * V_i + s_i * V_{i+1}

    approximating 1/2 sigma^2 S^2 d2V/dS2 + (r-q) S dV/dS - r V at node
    S_i = i*dS. Central differencing (2nd order) is used by default; at any
    node where it would produce a negative p_i or s_i, upwind differencing
    (1st order, unconditionally sign-safe) is substituted instead.
    """
    sig2i2 = (sigma ** 2) * (i ** 2)
    drift_i = (r - q) * i

    p_c = 0.5 * sig2i2 - 0.5 * drift_i
    s_c = 0.5 * sig2i2 + 0.5 * drift_i

    if r - q >= 0:
        p_u = 0.5 * sig2i2
        s_u = 0.5 * sig2i2 + drift_i
    else:
        p_u = 0.5 * sig2i2 - drift_i
        s_u = 0.5 * sig2i2

    safe = (p_c >= 0) & (s_c >= 0)
    p = np.where(safe, p_c, p_u)
    s = np.where(safe, s_c, s_u)
    q_coef = -(p + s) - r
    return p, q_coef, s


def _thomas_solve(a, b, c, d):
    """Thomas algorithm for a tridiagonal system. a[0] and c[-1] are not
    referenced (boundary contributions are assumed already folded into d)."""
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, n):
        m = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / m if k < n - 1 else 0.0
        dp[k] = (d[k] - a[k] * dp[k - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def _brennan_schwartz_solve(a, b, c, d, payoff):
    """Brennan & Schwartz (1977): same forward sweep as Thomas, but the
    backward substitution is projected onto V_i >= payoff_i at every node —
    solves the early-exercise linear complementarity problem exactly for a
    tridiagonal M-matrix system."""
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, n):
        m = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / m if k < n - 1 else 0.0
        dp[k] = (d[k] - a[k] * dp[k - 1]) / m
    x = np.empty(n)
    x[-1] = max(dp[-1], payoff[-1])
    for k in range(n - 2, -1, -1):
        x[k] = max(dp[k] - cp[k] * x[k + 1], payoff[k])
    return x


def _cell_average_payoff(S, dS, K, option_type, n_sub=21):
    """Average the terminal payoff over each grid cell [S_i - dS/2, S_i + dS/2]
    instead of sampling it at a single point. A kink (vanilla) sampled at one
    node introduces an O(dS) bias exactly where accuracy matters most —
    averaging removes it (Rannacher 1984; see also Tavella & Randall)."""
    offsets = ((np.arange(n_sub) + 0.5) / n_sub - 0.5) * dS
    samples = S[:, None] + offsets[None, :]
    return _payoff(samples, K, option_type).mean(axis=1)


# ---------------------------------------------------------------------------
# Finite-difference PDE solver
# ---------------------------------------------------------------------------

def crank_nicolson_price(
    S0, K, T, r, sigma, q=0.0,
    option_type="call", exercise="european",
    M=200, N=200, Smax_mult=4.0,
):
    """
    Solve the Black-Scholes PDE by finite differences.

      European: Crank-Nicolson with 2 Rannacher start-up steps.
      American: fully implicit + Brennan-Schwartz early-exercise projection.

    Returns a dict with:
      price   - interpolated value at (S0, t=0)
      S_grid  - the M+1 stock-price nodes
      V0      - option value V(S, t=0) across S_grid ("value vs spot" chart)
      payoff  - the terminal payoff across S_grid (same chart, dashed line)
      Smax, dS, dt - grid parameters used
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")
    if M < 10 or N < 10:
        raise ValueError("M and N must be at least 10 for a meaningful grid")

    theta = 1.0 if exercise == "american" else 0.5

    Smax = Smax_mult * max(S0, K)
    dS = Smax / M
    dt = T / N
    S = np.linspace(0.0, Smax, M + 1)
    i = np.arange(0, M + 1)

    p, q_coef, s = _fd_coefficients(i, sigma, r, q)

    # Cell-averaged payoff: used ONLY as the t=T terminal condition, to
    # reduce the O(dS) bias from sampling a kink at a single grid node.
    terminal = _cell_average_payoff(S, dS, K, option_type)
    terminal[0] = _payoff(S[0:1], K, option_type)[0]
    terminal[-1] = _payoff(S[-1:], K, option_type)[0]
    V = terminal.copy()  # V at t = T

    # True (non-averaged) intrinsic value: used as the American early-exercise
    # constraint at every subsequent step. Exercising gives you exactly
    # max(S-K,0) — a smoothed proxy would bias the free boundary.
    payoff = _payoff(S, K, option_type)

    for n in range(1, N + 1):
        # Rannacher: force the first two European steps fully implicit to
        # damp the payoff-kink oscillations Crank-Nicolson otherwise leaves.
        th = 1.0 if (exercise == "european" and n <= 2) else theta

        alpha = th * dt * p          # NEW-time (implicit) sub-diag weight
        beta = th * dt * q_coef      # NEW-time (implicit) diag weight
        gamma = th * dt * s          # NEW-time (implicit) super-diag weight
        alpha2 = (1 - th) * dt * p   # OLD-time (explicit) sub-diag weight
        beta2 = (1 - th) * dt * q_coef
        gamma2 = (1 - th) * dt * s

        a = -alpha[1:M]
        b = 1.0 - beta[1:M]
        c = -gamma[1:M]
        a2 = alpha2[1:M]
        b2 = 1.0 + beta2[1:M]
        c2 = gamma2[1:M]

        # n counts steps forward in tau = time-to-maturity (tau=0 is the
        # payoff at expiry, tau=T is today), so tau_new = n*dt IS the
        # time-to-maturity remaining at the level being solved for — no
        # further conversion needed.
        tau_new = n * dt

        if exercise == "american":
            # Immediate-exercise value at the domain edges (undiscounted:
            # you'd exercise now, not wait) — constant across time.
            if option_type == "call":
                V0_new, VM_new = 0.0, Smax - K
            else:
                V0_new, VM_new = K, 0.0
        else:
            # European: cannot exercise early, so use the discounted
            # asymptotic (deep ITM / OTM) value at the new time level.
            if option_type == "call":
                V0_new = 0.0
                VM_new = Smax * np.exp(-q * tau_new) - K * np.exp(-r * tau_new)
            else:
                V0_new = K * np.exp(-r * tau_new)
                VM_new = 0.0

        # V[0] and V[-1] here are still the OLD-time boundary values (from the
        # previous iteration / terminal condition), so this vectorized sum
        # already includes the explicit (old-time) boundary contribution
        # a2[0]*V0_old and c2[-1]*VM_old correctly. Only the NEW-time boundary
        # term needs to be folded in, since V0_new/VM_new are prescribed
        # (Dirichlet) rather than solved for.
        rhs = a2 * V[0:M - 1] + b2 * V[1:M] + c2 * V[2:M + 1]
        rhs[0] -= a[0] * V0_new
        rhs[-1] -= c[-1] * VM_new

        if exercise == "american":
            interior = _brennan_schwartz_solve(a, b, c, rhs, payoff[1:M])
        else:
            interior = _thomas_solve(a, b, c, rhs)

        V = np.empty(M + 1)
        V[0] = V0_new
        V[M] = VM_new
        V[1:M] = interior

    price = float(np.interp(S0, S, V))
    return {
        "price": price,
        "S_grid": S,
        "V0": V,
        "payoff": payoff,
        "Smax": Smax,
        "dS": dS,
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo_european(S0, K, T, r, sigma, q, option_type, n_paths, seed=None):
    """Exact terminal-distribution Monte Carlo with antithetic variates."""
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2
    z = rng.standard_normal(half)
    z = np.concatenate([z, -z])[:n_paths]
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoff = _payoff(ST, K, option_type)
    discounted = np.exp(-r * T) * payoff
    price = float(discounted.mean())
    se = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return {"price": price, "se": se, "ci_low": price - _Z95 * se, "ci_high": price + _Z95 * se}


def monte_carlo_american_lsm(S0, K, T, r, sigma, q, option_type, n_paths, n_steps, seed=None):
    """Longstaff-Schwartz (2001) Least-Squares Monte Carlo for American options."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    disc = np.exp(-r * dt)

    half = (n_paths + 1) // 2
    z = rng.standard_normal((half, n_steps))
    z = np.concatenate([z, -z], axis=0)[:n_paths]
    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    S = S0 * np.exp(log_paths)

    cash = _payoff(S[:, -1], K, option_type)
    for t in range(n_steps - 1, 0, -1):
        cash = cash * disc  # discount one step toward this exercise date
        immediate = _payoff(S[:, t], K, option_type)
        itm = immediate > 0
        if itm.sum() >= 5:  # need enough points for a stable quadratic fit
            x = S[itm, t]
            y = cash[itm]
            basis = np.column_stack([np.ones_like(x), x, x ** 2])
            coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
            continuation = basis @ coeffs
            exercise_now = immediate[itm] > continuation
            idx = np.where(itm)[0][exercise_now]
            cash[idx] = immediate[itm][exercise_now]
    cash = cash * disc  # final discount from t=1 back to t=0

    price = float(cash.mean())
    se = float(cash.std(ddof=1) / np.sqrt(n_paths))
    return {"price": price, "se": se, "ci_low": price - _Z95 * se, "ci_high": price + _Z95 * se}


# ---------------------------------------------------------------------------
# Binomial tree (American reference / benchmark)
# ---------------------------------------------------------------------------

def binomial_tree_american(S0, K, T, r, sigma, q, option_type, n_steps=1500):
    """Cox-Ross-Rubinstein (1979) binomial tree with an early-exercise check
    at every node. Standard textbook benchmark for American options, which
    have no closed-form price."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 1e-8), 1 - 1e-8)  # numerical safety net only

    j = np.arange(n_steps + 1)
    S_T = S0 * u ** (n_steps - j) * d ** j
    V = _payoff(S_T, K, option_type)

    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        S_node = S0 * u ** (step - j) * d ** j
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        V = np.maximum(V, _payoff(S_node, K, option_type))

    return float(V[0])


def validate_pde_inputs(S, K, T, r, sigma, q):
    """Reuse BlackScholesModel's bounds so the FD/MC engines and the
    closed-form engine never silently disagree on what's a valid input."""
    m = BlackScholesModel  # class-level constants only, no instantiation needed
    if S < m.MIN_STOCK_PRICE:
        raise ValueError(f"Stock price must be at least {m.MIN_STOCK_PRICE}")
    if K < m.MIN_STRIKE_PRICE:
        raise ValueError(f"Strike price must be at least {m.MIN_STRIKE_PRICE}")
    if T < m.MIN_TIME:
        raise ValueError(f"Time to maturity must be at least {m.MIN_TIME}")
    if sigma < m.MIN_VOLATILITY or sigma > m.MAX_VOLATILITY:
        raise ValueError(f"Volatility must be between {m.MIN_VOLATILITY} and {m.MAX_VOLATILITY}")
    if r < m.MIN_RATE or r > m.MAX_RATE:
        raise ValueError(f"Risk-free rate must be between {m.MIN_RATE} and {m.MAX_RATE}")
