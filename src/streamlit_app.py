"""
Options Pricing Calculator
Black-Scholes Model for European Options

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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from pricing_model import (
    BlackScholesModel, 
    calculate_implied_volatility, 
    quick_price,
    scenario_analysis
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Options Pricing Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 3rem; font-weight: 700; background: linear-gradient(120deg, #1f77b4, #ff7f0e); -webkit-background-clip: text; -webkit-fill-color: transparent; text-align: center; padding: 1rem 0; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.3rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .success-box { background-color: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .info-box { background-color: #e7f3ff; border-left: 4px solid #1f77b4; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .warning-box { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []

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


# HEADER
st.markdown('<div class="main-header"> Options Pricing Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Black-Scholes Model for European Options</div>', unsafe_allow_html=True)

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
    st.markdown("## Option Prices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### CALL OPTION")
        st.metric(
            "Call Price",
            f"${results['call_price']:.2f}",
            help="European Call Option Price"
        )
        st.metric(
            "Intrinsic Value",
            f"${results['call_intrinsic']:.2f}",
            help="Immediate exercise value"
        )
        st.metric(
            "Time Value",
            f"${results['call_time_value']:.2f}",
            help="Premium over intrinsic value"
        )
    
    with col2:
        st.markdown("### PUT OPTION")
        st.metric(
            "Put Price",
            f"${results['put_price']:.2f}",
            help="European Put Option Price"
        )
        st.metric(
            "Intrinsic Value",
            f"${results['put_intrinsic']:.2f}",
            help="Immediate exercise value"
        )
        st.metric(
            "Time Value",
            f"${results['put_time_value']:.2f}",
            help="Premium over intrinsic value"
        )
    
    # Put-Call Parity Check
    parity = results['parity_check']
    if parity['is_valid']:
        st.markdown('<div class="success-box"><b>Put-Call Parity Verified</b> - Calculations are mathematically consistent (Difference: $' + f"{parity['difference']:.4f}" + ')</div>', unsafe_allow_html=True)
    
    # SCENARIO ANALYSIS (replaces Trading Insights)
    st.markdown("## Scenario Analysis")
    st.caption("Stress-test option prices if the stock moves \u00b110% / \u00b120% / \u00b130% from today's price.")

    scenarios = scenario_analysis(
        S=params['stock_price'],
        K=params['strike_price'],
        T_days=params['days_to_expiry'],
        r_pct=params['risk_free_rate'],
        sigma_pct=params['volatility'],
        q_pct=params['dividend_yield'],
    )

    shock_labels = [f"{s['shock_pct']:+.0f}%" for s in scenarios]
    call_values = [s['call_price'] for s in scenarios]
    put_values = [s['put_price'] for s in scenarios]

    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Bar(
        x=shock_labels, y=call_values, name='Call Price',
        marker_color='#2ca02c', text=[f"${v:.2f}" for v in call_values], textposition='outside',
    ))
    fig_scenario.add_trace(go.Bar(
        x=shock_labels, y=put_values, name='Put Price',
        marker_color='#d62728', text=[f"${v:.2f}" for v in put_values], textposition='outside',
    ))
    fig_scenario.update_layout(
        barmode='group', xaxis_title='Stock Price Shock', yaxis_title='Option Price ($)',
        height=420, margin=dict(t=20, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_scenario, width='stretch')

    with st.expander("See underlying stock prices for each scenario"):
        scenario_df = pd.DataFrame(scenarios).rename(columns={
            'shock_pct': 'Shock (%)', 'stock_price': 'Stock Price ($)',
            'call_price': 'Call Price ($)', 'put_price': 'Put Price ($)',
        })
        st.dataframe(scenario_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # TABS
    tab1, tab3, tab4, tab5 = st.tabs([
        "Greeks Analysis",
        "Heat Map",
        "Advanced Analysis",
        "History and Export"
    ])
    
    # TAB 1: GREEKS
    with tab1:
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
                gauge={'axis': {'range': [0, 1]},
                      'bar': {'color': "darkgreen"},
                      'steps': [
                          {'range': [0, 0.3], 'color': "lightgray"},
                          {'range': [0.3, 0.7], 'color': "gray"},
                          {'range': [0.7, 1], 'color': "darkgray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75,
                                  'value': 0.5}}))
            fig_delta_call.update_layout(height=250)
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
                gauge={'axis': {'range': [0, 1]},
                      'bar': {'color': "darkred"},
                      'steps': [
                          {'range': [0, 0.3], 'color': "lightgray"},
                          {'range': [0.3, 0.7], 'color': "gray"},
                          {'range': [0.7, 1], 'color': "darkgray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75,
                                  'value': 0.5}}))
            fig_delta_put.update_layout(height=250)
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
    with tab3:
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
            height=450
        )
        st.plotly_chart(fig_heat, width='stretch')
    
    # TAB 4: ADVANCED ANALYSIS
    with tab4:
        st.markdown("### Advanced Analysis Tools")

        with st.expander("Understanding Volatility Smile"):
            st.markdown("""
            **Black-Scholes assumes constant volatility** across all strikes and maturities. The market disagrees.
    
            When you back out implied volatility from real market prices across different strikes,
            you get a U-shaped curve, not a flat line:
    
            - **Deep ITM options** → Higher IV
            - **ATM options** → Lowest IV
            - **Deep OTM options** → Higher IV
    
            This curvature is the **volatility smile**. For equity indices like the S&P 500,
            it becomes a **volatility skew** downside strikes carry systematically higher IV
            because traders pay a premium for crash protection.
    
            **What this means practically:**
            The 3D surface below shows IV computed independently at each strike using Black-Scholes inversion.
            If the model were perfect, the surface would be completely flat.
            The curvature you see is the market telling you Black-Scholes is incomplete.
            Real desks use Heston, SABR, or local volatility models to capture this structure.
            Understanding *why* the smile exists is more valuable than the smile itself.
            """)
    
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Implied Volatility Calculator")
            
            market_price_input = st.number_input(
                "Observed Market Price ($)",
                min_value=0.01,
                value=results['call_price'],
                step=0.01
            )
            
            option_type_iv = st.radio("Option Type", ['call', 'put'])
            
            if st.button("Calculate Implied Volatility"):
                with st.spinner("Calculating..."):
                    iv = calculate_implied_volatility(
                        market_price_input,
                        stock_price,
                        strike_price,
                        days_to_expiry / 365,
                        risk_free_rate / 100,
                        option_type_iv,
                        dividend_yield / 100
                    )
                    
                    if iv:
                        st.success(f"**Implied Volatility:** {iv*100:.2f}%")
                        st.metric("IV vs Current Vol", f"{iv*100:.2f}%", 
                                delta=f"{(iv*100 - volatility):.2f}%")
                    else:
                        st.error("Could not calculate IV. Check inputs.")
        
        with col2:
            st.markdown("#### Price Surface Heatmap")
            
            if st.button("Generate 3D Surface"):
                with st.spinner("Generating heatmap..."):
                    try:
                        sigma_range = np.linspace(max(0.01, volatility*0.5)/100, volatility*1.5/100, 15)
                        S_range_heat = np.linspace(stock_price*0.8, stock_price*1.2, 15)
                        
                        call_surface, put_surface = create_price_surface_heatmap(
                            stock_price, strike_price, days_to_expiry/365,
                            risk_free_rate/100, sigma_range, S_range_heat,
                            q=dividend_yield/100
                        )
                        
                        fig = go.Figure(data=[go.Surface(
                            x=S_range_heat,
                            y=sigma_range*100,
                            z=call_surface,
                            colorscale='Viridis',
                            name='Call Prices'
                        )])
                        
                        fig.update_layout(
                            title='Option Price Surface',
                            scene=dict(
                                xaxis_title='Stock Price ($)',
                                yaxis_title='Volatility (%)',
                                zaxis_title='Option Price ($)'
                            ),
                            height=600,
                            margin=dict(l=0, r=0, b=0, t=40)
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                    except Exception as e:
                        st.error(f"Error generating surface: {str(e)}")
    
    # TAB 5: HISTORY & EXPORT
    with tab5:
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
        <h3>Options Pricing Analyzer</h3>
        <p><b>Black-Scholes Model • European Options</b></p>
        <p style='font-size: 0.9em; margin-top: 1rem;'>
            Created by <b>Nattawut Boonnoon</b><br>
            <a href="https://www.linkedin.com/in/nattawut-bn" target="_blank" style="color: #0077b5; text-decoration: none;">
                 LinkedIn Profile (Click)
            </a> • 
            <a href="https://github.com/Nattawut30" target="_blank" style="color: #0077b5; text-decoration: none;">
                 GitHub Profile (Click)
            </a>
        </p>
        <p style='font-size: 0.8em; color: #999; margin-top: 1rem;'>
            *** For quantitative analysis and hedging strategy. ***
        </p>
    </div>
""", unsafe_allow_html=True)
