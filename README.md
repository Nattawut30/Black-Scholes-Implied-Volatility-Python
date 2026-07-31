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

This is my options pricing project that combines educational clarity with real-world numerical methods. It prices European options in closed form (Black-Scholes) and prices both European and American options numerically via finite differences and Monte Carlo, cross-checking all three methods against each other on every calculation. It also includes a Greeks dashboard, a price-surface heat map, and calculation history/export.

# <p align="center">What is the Black-Scholes model? <p/>
The Black-Scholes, or Black-Scholes-Merton model is a mathematical model that describes the trends of a financial market, including derivative investment instruments. The formula and model are named after the economists *Fischer Black* and *Myron Scholes*. Occasionally, attribution is also awarded to *Robert C. Merton*, who was the first to write an academic paper on the topic.

The model's fundamental objective is to hedge the option by purchasing and selling the underlying asset in a precise pattern to remove risk. This type of hedging is known as "constantly modified delta hedging" and forms the foundation of more complex hedging strategies utilized by investment firms and hedge funds.

Call Options Price:
$$
C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

Put Options Price:
$$
P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)
$$

Where:
$$
d_1 = \frac{\ln(S_0/K) + \left(r - q + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
$$

The PDE:
$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q)S\frac{\partial V}{\partial S} - rV = 0
$$

Parameters:

$S_0$ = Current stock price
$K$ = Strike price
$N(x)$ = Cumulative normal distribution

# <p align="center">Acknowledgments<p/>

**Dependencies:**
- `streamlit` - Webapp framework
- `pandas` - Data manipulation
- `numpy` - Numerical calculations
- `plotly` - Interactive charts
- `pyarrow` - Data processing

**Academic Papers & References:**
- Black, F., & Scholes, M. (1973). *"The Pricing of Options and Corporate Liabilities"*
- Merton, R. C. (1973). *"Theory of Rational Option Pricing"*
- Brennan, M. J., & Schwartz, E. S. (1977). *"The Valuation of American Put Options"*
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). *"Option Pricing: A Simplified Approach"*
- Rannacher, R. (1984). *"Finite Element Solution of Diffusion Problems with Irregular Data"*
- Longstaff, F. A., & Schwartz, E. S. (2001). *"Valuing American Options by Simulation: A Simple Least-Squares Approach"*
- Paolucci, R., Quant Guild., (2026). *"Numerical methods design cross-checked."*
