# <p align="center"> Python: Black-Scholes PDE Solver <p/>
<br>**Nattawut Boonnoon**<br/>
- LinkedIn: www.linkedin.com/in/nattawut-bn
- Email: nattawut.boonnoon@hotmail.com

***Overview***
- 
[![Keep Streamlit App Awake](https://github.com/Nattawut30/Black-Scholes-Implied-Volatility-Python/actions/workflows/keep-alive.yml/badge.svg)](https://github.com/Nattawut30/Black-Scholes-Implied-Volatility-Python/actions/workflows/keep-alive.yml)
![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Ready-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)

Here: https://nattawut-blsm.streamlit.app <br>

Presentation Slides: [Click](https://gamma.app/docs/Black-Scholes-Implied-Volatility-By-Nattawut-B-la8qpqrj0j6scxi)

Updated: July 2026 <br>
- Added a finite-difference PDE solver (Crank-Nicolson + Rannacher start-up for European; fully implicit + Brennan-Schwartz for American early exercise), a Monte Carlo engine (exact terminal simulation for European, Longstaff-Schwartz for American), and a Cox-Ross-Rubinstein binomial tree as an independent American reference.
- Added American option support (Call/Put) alongside the existing closed-form European model.
- Resolved fault memory issues related to SciPy and Streamlit.
- Updated the project on the master/main branch and resolved CI problems.
- Migrated dependency management to Poetry.

This is my options pricing project that combines educational clarity with real-world numerical methods. It prices European options in closed form (Black-Scholes) and prices both European and American options numerically via finite differences and Monte Carlo, cross-checking all three methods against each other on every calculation. It also includes a Greeks dashboard, a price-surface heat map, and calculation history/export.

# <p align="center">What is the Black-Scholes model? <p/>
The Black-Scholes, or Black-Scholes-Merton model is a mathematical model that describes the trends of a financial market, including derivative investment instruments. The formula and model are named after the economists *Fischer Black* and *Myron Scholes*. Occasionally, attribution is also awarded to *Robert C. Merton*, who was the first to write an academic paper on the topic.

The model's fundamental objective is to hedge the option by purchasing and selling the underlying asset in a precise pattern to remove risk. This type of hedging is known as "constantly modified delta hedging" and forms the foundation of more complex hedging strategies utilized by investment firms and hedge funds.

Call Options Price:
`````bash
C = S₀·N(d₁) - K·e^(-rT)·N(d₂)
`````
Put Options Price:
`````bash
P = K·e^(-rT)·N(-d₂) - S₀·N(-d₁)
`````
Where:
`````bash
d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
`````
Parameters:

S₀ = Current stock price <br>
K = Strike price <br>
T = Time to expiration (years) <br>
r = Risk-free interest rate <br>
σ = Volatility (annual) <br>
N(x) = Cumulative normal distribution <br>

Closed-form pricing like this only exists for **European** options (exercisable at expiry only). **American** options (exercisable any time up to expiry) have no closed-form price, so this project solves the same Black-Scholes PDE numerically instead:

`````bash
∂V/∂t + ½σ²S²(∂²V/∂S²) + (r-q)S(∂V/∂S) - rV = 0
`````

subject to the terminal payoff at expiry and, for American options, the constraint that the option is always worth at least its immediate exercise value. Three independent methods are computed for every price shown in the app: finite differences, Monte Carlo, and a closed-form/binomial-tree reference. so the numbers can be cross-checked against each other rather than trusted blindly.

# <p align="center">Acknowledgments<p/>

**Dependencies:**
- `streamlit` - Webapp framework
- `pandas` - Data manipulation
- `numpy` - Numerical calculations
- `plotly` - Interactive charts
- `pyarrow` - Data processing

**Academic Papers:**
- Black, F., & Scholes, M. (1973). *"The Pricing of Options and Corporate Liabilities"*
- Merton, R. C. (1973). *"Theory of Rational Option Pricing"*
- Brennan, M. J., & Schwartz, E. S. (1977). *"The Valuation of American Put Options"*
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). *"Option Pricing: A Simplified Approach"*
- Rannacher, R. (1984). *"Finite Element Solution of Diffusion Problems with Irregular Data"*
- Longstaff, F. A., & Schwartz, E. S. (2001). *"Valuing American Options by Simulation: A Simple Least-Squares Approach"*
- Paolucci, R., Quant Guild., (2026). *"Numerical methods design cross-checked: Quantitative Researcher"*
