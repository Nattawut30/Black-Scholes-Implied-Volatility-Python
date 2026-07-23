"""
Black-Scholes PDE Solver
Finite Differences, Monte Carlo & Closed-Form — European & American Options

Created By: Nattawut Boonnoon
LinkedIn: www.linkedin.com/in/nattawut-bn
GitHub: https://github.com/Nattawut30
Email: nattawut.boonnoon@hotmail.com
Location: Bangkok, Thailand


Thank you!


"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from pricing_model import (
    BlackScholesModel, 
    quick_price,
)
from pde_model import (
    crank_nicolson_price,
    monte_carlo_european,
    monte_carlo_american_lsm,
    binomial_tree_american,
    validate_pde_inputs,
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Black-Scholes PDE Solver",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 3rem; font-weight: 700; background: linear-gradient(120deg, #89b4fa, #cba6f7); -webkit-background-clip: text; -webkit-fill-color: transparent; text-align: center; padding: 1rem 0; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.3rem; color: #a6adc8; text-align: center; margin-bottom: 2rem; }
    .stMetric { background-color: #313244; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(17,17,27,0.4); }
    .success-box { background-color: #313244; border-left: 4px solid #a6e3a1; color: #cdd6f4; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .info-box { background-color: #313244; border-left: 4px solid #89b4fa; color: #cdd6f4; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .warning-box { background-color: #313244; border-left: 4px solid #fab387; color: #cdd6f4; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []

# Catppuccin Mocha palette for Plotly charts (Plotly figures render in their own canvas
# and do NOT automatically pick up Streamlit's theme, so we apply it explicitly here).
MOCHA_CHART_LAYOUT = dict(
    paper_bgcolor='#1e1e2e',
    plot_bgcolor='#1e1e2e',
    font=dict(color='#cdd6f4'),
)

def create_price_surface_heatmap(S, K, T, r, sigma_range, S_range, q=0.0):
    call_prices = np.zeros((len(sigma_range), len(S_range)))
    put_prices = np.zeros((len(sigma_range), len(S_range)))
    
    for i, sigma in enumerate(sigma_range):
        for j, stock_price in enumerate(S_range):
            try:
                model = BlackScholesModel(stock_price, K, T, r, sigma, dividend_yield=q)
                call_prices[i, j] = model.call_price()
                put_prices[i, j] = model.put_price()
            except:
                call_prices[i, j] = np.nan
                put_prices[i, j] = np.nan
    return call_prices, put_prices
    

def export_to_csv(data, filename):
    """Export calculation results to CSV"""
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    return csv


def _closed_form_price(S, K, T, r, sigma, q, option_type):
    model = BlackScholesModel(S, K, T, r, sigma, q)
    return model.call_price() if option_type == "call" else model.put_price()


@st.cache_data(show_spinner=False)
def _cached_fd(S, K, T, r, sigma, q, option_type, exercise, M, N):
    return crank_nicolson_price(S, K, T, r, sigma, q, option_type=option_type,
                                 exercise=exercise, M=M, N=N)


@st.cache_data(show_spinner=False)
def _cached_mc_european(S, K, T, r, sigma, q, option_type, n_paths, seed):
    return monte_carlo_european(S, K, T, r, sigma, q, option_type, n_paths, seed=seed)


@st.cache_data(show_spinner=False)
def _cached_mc_american(S, K, T, r, sigma, q, option_type, n_paths, n_steps, seed):
    return monte_carlo_american_lsm(S, K, T, r, sigma, q, option_type, n_paths, n_steps, seed=seed)


@st.cache_data(show_spinner=False)
def _cached_binomial(S, K, T, r, sigma, q, option_type, n_steps):
    return binomial_tree_american(S, K, T, r, sigma, q, option_type, n_steps=n_steps)


# HEADER
st.markdown('<div class="main-header"> Black-Scholes PDE Solver</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Finite Differences &bull; Monte Carlo &bull; Closed-Form; European &amp; American Options</div>', unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("⚙️ Configuration")

# Reset button
if st.sidebar.button("Reset All Values", width='stretch'):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("### Option Parameters")

stock_price = st.sidebar.number_input(
    "Current Stock Price ($)",
    min_value=0.01,
    max_value=100000.0,
    value=100.0,
    step=1.0,
    format="%.2f"
)

strike_price = st.sidebar.number_input(
    "Strike Price ($)",
    min_value=0.01,
    max_value=100000.0,
    value=100.0,
    step=1.0,
    format="%.2f"
)

moneyness = stock_price / strike_price
if moneyness > 1.05:
    st.sidebar.info("In-The-Money (ITM) for Calls")
elif moneyness < 0.95:
    st.sidebar.info("In-The-Money (ITM) for Puts")
else:
    st.sidebar.info("At-The-Money (ATM)")

st.sidebar.markdown("### Contract")

option_style = st.sidebar.radio(
    "Exercise Style",
    ["European", "American"],
    horizontal=True,
    help="European: exercisable only at expiry (closed-form Black-Scholes exists). "
         "American: exercisable any time up to expiry (no closed form — priced by "
         "finite differences and Monte Carlo, cross-checked against a binomial tree)."
)

option_type_selected = st.sidebar.radio(
    "Option Type",
    ["Call", "Put"],
    horizontal=True,
    help="Which contract the PDE / Monte Carlo / reference cards below are priced for."
)

st.sidebar.markdown("### Time & Market")

days_to_expiry = st.sidebar.slider(
    "Days to Expiration",
    min_value=1,
    max_value=730,
    value=30,
    help="Trading days until option expires"
)

volatility = st.sidebar.slider(
    "Volatility (% Annual)",
    min_value=1.0,
    max_value=200.0,
    value=25.0,
    step=0.5,
    help="Standard deviation of returns"
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (% Annual)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1,
    help="Treasury rate or LIBOR"
)

dividend_yield = st.sidebar.slider(
    "Dividend Yield (% Annual)",
    min_value=0.0,
    max_value=25.0,
    value=0.0,
    step=0.1,
    help="Continuous dividend yield. Use 0.0 for non-dividend paying stocks like most tech companies."
)

st.sidebar.markdown("### Finite-Difference Grid")
st.sidebar.caption("European: Crank-Nicolson (+ Rannacher start-up). American: fully implicit + Brennan-Schwartz.")

fd_S_steps = st.sidebar.slider(
    "S-steps (M)", min_value=50, max_value=400, value=300, step=10,
    help="Number of stock-price grid points. Higher = more accurate, slower."
)
fd_t_steps = st.sidebar.slider(
    "t-steps (N)", min_value=50, max_value=400, value=300, step=10,
    help="Number of time-grid points. Higher = more accurate, slower."
)

st.sidebar.markdown("### Monte Carlo")
mc_paths = st.sidebar.slider(
    "Paths", min_value=5000, max_value=200000, value=50000, step=5000,
    help="More paths tighten the 95% confidence interval (error shrinks like 1/sqrt(paths))."
)
if option_style == "American":
    mc_steps = st.sidebar.slider(
        "Exercise dates (steps)", min_value=10, max_value=150, value=50, step=10,
        help="Longstaff-Schwartz needs discrete exercise dates. More steps = closer to "
             "true continuous exercise, slower to compute."
    )
    if mc_paths > 50000 or mc_steps > 100:
        st.sidebar.caption(":gray[American Monte Carlo (Longstaff-Schwartz) is much more "
                            "compute-heavy than European MC, very high paths/steps may take "
                            "a few seconds.]")
else:
    mc_steps = None  # European MC is an exact terminal-distribution draw; no path stepping needed

st.sidebar.markdown("---")

calculate_btn = st.sidebar.button(
    "Calculate Options",
    type="primary",
    width='stretch'
)

# MAIN CALCULATION
if calculate_btn or 'results' in st.session_state:
    try:
        results = quick_price(
            S=stock_price,
            K=strike_price,
            T_days=days_to_expiry,
            r_pct=risk_free_rate,
            sigma_pct=volatility,
            q_pct=dividend_yield
        )

        # --- PDE / Monte Carlo / reference engines (rebrand: requirements 3 & 4) ---
        S_, K_, T_, r_, sig_, q_ = (
            stock_price, strike_price, days_to_expiry / 365,
            risk_free_rate / 100, volatility / 100, dividend_yield / 100,
        )
        opt_lower = option_type_selected.lower()
        exercise_lower = option_style.lower()
        validate_pde_inputs(S_, K_, T_, r_, sig_, q_)

        fd_result = _cached_fd(S_, K_, T_, r_, sig_, q_, opt_lower, exercise_lower,
                                fd_S_steps, fd_t_steps)

        if exercise_lower == "european":
            mc_result = _cached_mc_european(S_, K_, T_, r_, sig_, q_, opt_lower, mc_paths, 42)
            reference_price = _closed_form_price(S_, K_, T_, r_, sig_, q_, opt_lower)
            reference_label = "Closed-Form (Analytic)"
        else:
            mc_result = _cached_mc_american(S_, K_, T_, r_, sig_, q_, opt_lower, mc_paths, mc_steps, 42)
            reference_price = _cached_binomial(S_, K_, T_, r_, sig_, q_, opt_lower, 1500)
            reference_label = "Reference (Binomial Tree, N=1500)"

        st.session_state.pde_results = {
            'fd': fd_result,
            'mc': mc_result,
            'reference_price': reference_price,
            'reference_label': reference_label,
            'option_type': opt_lower,
            'exercise': exercise_lower,
        }

        st.session_state.results = results
        st.session_state.params = {
            'stock_price': stock_price,
            'strike_price': strike_price,
            'days_to_expiry': days_to_expiry,
            'volatility': volatility,
            'risk_free_rate': risk_free_rate,
            'dividend_yield': dividend_yield
        }
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.calculation_history.append({
            'timestamp': timestamp,
            'S': stock_price,
            'K': strike_price,
            'T': days_to_expiry,
            'sigma': volatility,
            'r': risk_free_rate,
            'call': results['call_price'],
            'put': results['put_price']
        })
        
        if len(st.session_state.calculation_history) > 10:
            st.session_state.calculation_history.pop(0)
        
    except Exception as e:
        st.error(f"Calculation Error: {str(e)}")

# DISPLAY RESULTS
if 'results' in st.session_state:
    results = st.session_state.results
    params = st.session_state.params
    greeks = results['greeks']
    
    # OPTION PRICES - SIDE BY SIDE
    pde = st.session_state.pde_results
    fd = pde['fd']
    mc = pde['mc']
    ref_price = pde['reference_price']
    ref_label = pde['reference_label']
    opt_lbl = pde['option_type']
    is_american = pde['exercise'] == 'american'

    # GOVERNING EQUATION & CONDITIONS
    st.markdown("## Governing Equation & Conditions")
    st.latex(r"\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 "
             r"\frac{\partial^2 V}{\partial S^2} + (r-q) S \frac{\partial V}{\partial S} - rV = 0")

    if opt_lbl == 'call':
        terminal_tex = r"V(S,T)=\max(S-K,\,0)"
    else:
        terminal_tex = r"V(S,T)=\max(K-S,\,0)"

    if is_american:
        boundary_tex = r"V(S,t)\ \ge\ \text{payoff}(S)\ \ \text{for all } t \le T \quad \text{(early exercise)}"
    elif opt_lbl == 'call':
        boundary_tex = r"V(0,t)=0,\quad V(S_{\max},t)=S_{\max}e^{-q\tau}-Ke^{-r\tau}"
    else:
        boundary_tex = r"V(0,t)=Ke^{-r\tau},\quad V(S_{\max},t)=0"

    col_t, col_b = st.columns(2)
    with col_t:
        st.caption("TERMINAL")
        st.latex(terminal_tex)
    with col_b:
        st.caption("BOUNDARY" + ("" if is_american else " (\u03c4 = T \u2212 t)"))
        st.latex(boundary_tex)

    st.caption(f"Solving for the **American {opt_lbl}**" if is_american
               else f"Solving for the **European {opt_lbl}**")

    # FD / MC / REFERENCE COMPARISON CARDS
    st.markdown("## Numerical Solution")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**FINITE DIFFERENCE**")
        st.metric("Price", f"${fd['price']:.4f}",
                  help="Crank-Nicolson (European) or fully-implicit + Brennan-Schwartz (American).")
        st.caption(f"{'implicit + Brennan-Schwartz' if is_american else 'Crank-Nicolson'} \u00b7 {fd_S_steps}\u00d7{fd_t_steps} grid")
    with c2:
        st.markdown("**MONTE CARLO**")
        st.metric("Price", f"${mc['price']:.4f}",
                  help="95% confidence interval from the simulation's standard error.")
        st.caption(f"95% CI [{mc['ci_low']:.4f}, {mc['ci_high']:.4f}]")
    with c3:
        st.markdown("**" + ref_label.split(" (")[0].upper() + "**")
        st.metric("Price", f"${ref_price:.4f}" if ref_price is not None else "n/a",
                  help=ref_label)
        st.caption(ref_label)
    with c4:
        diff = fd['price'] - ref_price
        rel = (diff / ref_price * 100) if ref_price and abs(ref_price) > 1e-6 else float('nan')
        outside_ci = not (mc['ci_low'] <= fd['price'] <= mc['ci_high'])
        st.markdown("**FD \u2212 REFERENCE**")
        st.metric("Difference", f"${diff:+.4f}",
                   delta=f"{rel:+.2f}%" if rel == rel else None,
                   help="FD price minus the reference price (closed-form for European, "
                        "binomial tree for American).")
        st.caption(":orange[outside FD-MC 95% CI]" if outside_ci else ":green[within FD-MC 95% CI]")

    st.markdown("---")

    # CHART 1: VALUE VS SPOT AT t=0
    st.markdown("## Value vs Spot at t = 0")
    st.caption("The finite-difference solution V(S, 0) across the whole grid, against the terminal payoff.")

    fig_vs = go.Figure()
    fig_vs.add_trace(go.Scatter(
        x=fd['S_grid'], y=fd['payoff'], name='Payoff', mode='lines',
        line=dict(color='#f38ba8', dash='dot'),
    ))
    fig_vs.add_trace(go.Scatter(
        x=fd['S_grid'], y=fd['V0'], name='V(S, 0)', mode='lines',
        line=dict(color='#89b4fa', width=2.5),
    ))
    fig_vs.add_trace(go.Scatter(
        x=[stock_price], y=[fd['price']], name=f'S\u2080 = {stock_price:.2f}',
        mode='markers', marker=dict(color='#fab387', size=11, symbol='diamond'),
    ))
    plot_max_x = min(fd['Smax'], strike_price * 2.5, stock_price * 2.5) or fd['Smax']
    fig_vs.update_layout(
        xaxis_title='Spot Price S', yaxis_title='Option Value ($)',
        xaxis_range=[0, plot_max_x],
        height=420, margin=dict(t=20, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        **MOCHA_CHART_LAYOUT,
    )
    fig_vs.update_xaxes(gridcolor='#313244')
    fig_vs.update_yaxes(gridcolor='#313244')
    st.plotly_chart(fig_vs, width='stretch')

    # CHART 2: FD vs MC vs REFERENCE
    st.markdown("## FD vs MC vs " + ref_label.split(" (")[0])
    st.caption("Three independent pricing methods, cross-checked against each other.")

    fig_cmp = go.Figure()
    bar_names = ['Finite Difference', 'Monte Carlo', ref_label.split(" (")[0]]
    bar_values = [fd['price'], mc['price'], ref_price]
    bar_colors = ['#89b4fa', '#a6e3a1', '#cba6f7']
    fig_cmp.add_trace(go.Bar(
        x=bar_names, y=bar_values, marker_color=bar_colors,
        text=[f"${v:.4f}" for v in bar_values], textposition='outside',
        error_y=dict(
            type='data', symmetric=False,
            array=[0, mc['ci_high'] - mc['price'], 0],
            arrayminus=[0, mc['price'] - mc['ci_low'], 0],
            visible=True, color='#a6adc8',
        ),
    ))
    fig_cmp.update_layout(
        yaxis_title='Option Price ($)', height=420, margin=dict(t=20, b=20),
        showlegend=False, **MOCHA_CHART_LAYOUT,
    )
    fig_cmp.update_xaxes(gridcolor='#313244')
    fig_cmp.update_yaxes(gridcolor='#313244')
    st.plotly_chart(fig_cmp, width='stretch')

    st.markdown("---")

    # PRICE SURFACE HEATMAP (moved out of the old Advanced Analysis tab,
    # now sits right below the FD vs MC vs Reference comparison)
    st.markdown("## Price Surface Heatmap")
    st.caption("Option price across a range of stock prices and volatilities, "
               "using the closed-form European model.")

    if st.button("Generate 3D Surface"):
        with st.spinner("Generating heatmap..."):
            try:
                sigma_range = np.linspace(max(0.01, volatility * 0.5) / 100, volatility * 1.5 / 100, 15)
                S_range_heat = np.linspace(stock_price * 0.8, stock_price * 1.2, 15)

                call_surface, put_surface = create_price_surface_heatmap(
                    stock_price, strike_price, days_to_expiry / 365,
                    risk_free_rate / 100, sigma_range, S_range_heat,
                    q=dividend_yield / 100
                )

                fig = go.Figure(data=[go.Surface(
                    x=S_range_heat,
                    y=sigma_range * 100,
                    z=call_surface,
                    colorscale='Viridis',
                    name='Call Prices'
                )])

                fig.update_layout(
                    title='Option Price Surface',
                    scene=dict(
                        xaxis_title='Stock Price ($)',
                        yaxis_title='Volatility (%)',
                        zaxis_title='Option Price ($)',
                        xaxis=dict(backgroundcolor='#1e1e2e', gridcolor='#45475a', color='#cdd6f4'),
                        yaxis=dict(backgroundcolor='#1e1e2e', gridcolor='#45475a', color='#cdd6f4'),
                        zaxis=dict(backgroundcolor='#1e1e2e', gridcolor='#45475a', color='#cdd6f4'),
                    ),
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=40),
                    **MOCHA_CHART_LAYOUT,
                )

                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error(f"Error generating surface: {str(e)}")

    st.markdown("---")
    
    # TABS
    st.caption("The tabs below (Greeks, Heat Map, History) run on the "
               "closed-form **European** Black-Scholes model regardless of the Exercise Style "
               "selected in the sidebar \u2014 American options have no closed-form Greeks, "
               "since those require differentiating the numerical PDE solution.")
    tab_greeks, tab_heatmap, tab_history = st.tabs([
        "Greeks Analysis",
        "Heat Map",
        "History and Export"
    ])
    
    # TAB 1: GREEKS
    with tab_greeks:
        st.markdown("### Option Greeks - Risk Metrics")
        
        greeks = results['greeks']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Call Option Greeks")
            
            greeks_data_call = {
                'Greek': ['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)', 'Rho (ρ)'],
                'Value': [
                    f"{greeks['call_delta']:.4f}",
                    f"{greeks['call_gamma']:.4f}",
                    f"{greeks['call_theta']:.4f}",
                    f"{greeks['call_vega']:.4f}",
                    f"{greeks['call_rho']:.4f}"
                ],
                'Meaning': [
                    f'${greeks["call_delta"]:.2f} per $1 stock move',
                    f'Delta changes by {greeks["call_gamma"]:.4f} per $1 move',
                    f'Loses ${abs(greeks["call_theta"]):.2f} per day',
                    f'${greeks["call_vega"]:.2f} per 1% volatility change',
                    f'${greeks["call_rho"]:.2f} per 1% rate change'
                ]
            }
            
            df_call = pd.DataFrame(greeks_data_call)
            st.dataframe(df_call, width='stretch', hide_index=True)
            
            fig_delta_call = go.Figure(go.Indicator(
                mode="gauge+number",
                value=greeks['call_delta'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Call Delta"},
                number={'font': {'color': '#cdd6f4'}},
                gauge={'axis': {'range': [0, 1], 'tickcolor': '#cdd6f4'},
                      'bar': {'color': "#a6e3a1"},
                      'bgcolor': '#1e1e2e',
                      'steps': [
                          {'range': [0, 0.3], 'color': "#313244"},
                          {'range': [0.3, 0.7], 'color': "#45475a"},
                          {'range': [0.7, 1], 'color': "#585b70"}],
                      'threshold': {'line': {'color': "#fab387", 'width': 4},
                                  'thickness': 0.75,
                                  'value': 0.5}}))
            fig_delta_call.update_layout(height=250, **MOCHA_CHART_LAYOUT)
            st.plotly_chart(fig_delta_call, width='stretch')
        
        with col2:
            st.markdown("#### Put Option Greeks")
            
            greeks_data_put = {
                'Greek': ['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)', 'Rho (ρ)'],
                'Value': [
                    f"{greeks['put_delta']:.4f}",
                    f"{greeks['put_gamma']:.4f}",
                    f"{greeks['put_theta']:.4f}",
                    f"{greeks['put_vega']:.4f}",
                    f"{greeks['put_rho']:.4f}"
                ],
                'Meaning': [
                    f'${greeks["put_delta"]:.2f} per $1 stock move',
                    f'Delta changes by {greeks["put_gamma"]:.4f} per $1 move',
                    f'Loses ${abs(greeks["put_theta"]):.2f} per day',
                    f'${greeks["put_vega"]:.2f} per 1% volatility change',
                    f'${greeks["put_rho"]:.2f} per 1% rate change'
                ]
            }
            
            df_put = pd.DataFrame(greeks_data_put)
            st.dataframe(df_put, width='stretch', hide_index=True)
            
            fig_delta_put = go.Figure(go.Indicator(
                mode="gauge+number",
                value=abs(greeks['put_delta']),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Put Delta (Absolute)"},
                number={'font': {'color': '#cdd6f4'}},
                gauge={'axis': {'range': [0, 1], 'tickcolor': '#cdd6f4'},
                      'bar': {'color': "#f38ba8"},
                      'bgcolor': '#1e1e2e',
                      'steps': [
                          {'range': [0, 0.3], 'color': "#313244"},
                          {'range': [0.3, 0.7], 'color': "#45475a"},
                          {'range': [0.7, 1], 'color': "#585b70"}],
                      'threshold': {'line': {'color': "#fab387", 'width': 4},
                                  'thickness': 0.75,
                                  'value': 0.5}}))
            fig_delta_put.update_layout(height=250, **MOCHA_CHART_LAYOUT)
            st.plotly_chart(fig_delta_put, width='stretch')
        
        with st.expander("Understanding Greeks"):
            st.markdown("""
            ### What Each Greek Tells You:
            
            **Delta (Δ)** - Directional Risk
            - Measures price change per $1 stock move
            - Call: 0 to 1 | Put: -1 to 0
            
            **Gamma (Γ)** - Delta Risk
            - How fast Delta changes
            - Important for hedging
            
            **Theta (Θ)** - Time Decay
            - Value lost each day
            - Always negative for long positions
            
            **Vega (ν)** - Volatility Risk
            - Price change per 1% volatility move
            - Higher for longer-dated options
            
            **Rho (ρ)** - Interest Rate Risk
            - Price change per 1% rate change
            - Usually smallest Greek
            """)
        st.markdown("---")
        st.markdown("### Second-Order Greeks")
        st.caption("Used by professional options desks for advanced hedging.")
        col_v, col_c = st.columns(2)
        with col_v:
            st.metric(
                "Vanna",
                f"{greeks['vanna']:.6f}",
                help="∂Delta/∂σ — How delta shifts when volatility moves. Used to manage delta exposure after a vol spike."
            )
        with col_c:
            st.metric(
                "Call Charm",
                f"{greeks['call_charm']:.6f}",
                help="−∂Delta_Call/∂t — Declining of Call Delta Per day"
            )
            st.metric(
                "Put Charm",
                f"{greeks['put_charm']:.6f}",
                help="−∂Delta_Put/∂t — Changing of Put Delta Per day"
            )
    
    # TAB 3: SIMPLE HEATMAP
    with tab_heatmap:
        st.markdown("### Heatmap")
        
        S_matrix = np.linspace(stock_price * 0.9, stock_price * 1.1, 7)
        vol_matrix = np.linspace(max(5.0, volatility - 15), volatility + 15, 7)
        
        opt_type = st.radio("Heatmap Choices:", ["Call Option", "Put Option"], horizontal=True)
        
        heatmap_data = []
        for v in vol_matrix:
            row = []
            for s in S_matrix:
                model = BlackScholesModel(s, strike_price, days_to_expiry/365, risk_free_rate/100, v/100, dividend_yield/100)
                price = model.call_price() if opt_type == "Call Option" else model.put_price()
                row.append(price)
            heatmap_data.append(row)
            
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"${x:.1f}" for x in S_matrix],
            y=[f"{y:.1f}%" for y in vol_matrix],
            colorscale='YlGnBu' if opt_type == "Call Option" else 'OrRd',
            text=np.round(heatmap_data, 2),
            texttemplate="%{text}",
            hoverinfo="z"
        ))
        
        fig_heat.update_layout(
            xaxis_title="ราคาสินค้าอ้างอิง (Stock Price)",
            yaxis_title="ความผันผวน (Volatility %)",
            height=450,
            **MOCHA_CHART_LAYOUT,
        )
        st.plotly_chart(fig_heat, width='stretch')
    
    # TAB 3: HISTORY & EXPORT
    with tab_history:
        st.markdown("### Calculation History")
        
        if st.session_state.calculation_history:
            history_df = pd.DataFrame(st.session_state.calculation_history)
            st.dataframe(history_df, width='stretch')
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = export_to_csv(st.session_state.calculation_history, 'history.csv')
                st.download_button(
                    "Download History (CSV)",
                    csv,
                    "options_history.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            with col2:
                current_export = {
                    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Stock_Price': [stock_price],
                    'Strike_Price': [strike_price],
                    'Days_to_Expiry': [days_to_expiry],
                    'Volatility': [volatility],
                    'Risk_Free_Rate': [risk_free_rate],
                    'Call_Price': [results['call_price']],
                    'Put_Price': [results['put_price']],
                    'Call_Delta': [greeks['call_delta']],
                    'Put_Delta': [greeks['put_delta']],
                    'Gamma': [greeks['call_gamma']],
                    'Vega': [greeks['call_vega']],
                    'Call_Theta': [greeks['call_theta']],
                    'Put_Theta': [greeks['put_theta']]
                }
                
                csv_current = export_to_csv(current_export, 'current.csv')
                st.download_button(
                    "Download Current Results",
                    csv_current,
                    "current_calculation.csv",
                    "text/csv",
                    key='download-current'
                )
        else:
            st.info("No calculations in history yet. Run a calculation to see it here.")

# FOOTER END GAME 
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <h3>Black-Scholes PDE Solver</h3>
        <p><b>Finite Differences • Monte Carlo • Closed-Form • European &amp; American Options</b></p>
        <p style='font-size: 0.9em; margin-top: 1rem;'>
            Created by <b>Nattawut Boonnoon</b><br>
            <a href="https://www.linkedin.com/in/nattawut-bn" target="_blank" style="color: #0077b5; text-decoration: none;">
                 LinkedIn Profile (Click)
            </a> • 
            <a href="https://github.com/Nattawut30" target="_blank" style="color: #0077b5; text-decoration: none;">
                 GitHub Profile (Click)
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)
