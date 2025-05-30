import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from option_pricing.black_scholes_model import bs_portfolio_analysis as bsa
from option_pricing.binomial_model import binomial_portfolio_analysis as bpa
from option_pricing.monte_carlo import mc_portfolio_analysis as mca

# Set a fixed random seed for reproducibility
np.random.seed(42)

st.set_page_config(page_title="Option Pricing & Portfolio Risk App", layout="wide")

st.markdown("""
<style>
    .main, .stApp, .css-18e3th9 {
        background-color: #1e1e1e;
    }
    .title-conference {
        font-size: 2.4rem;
        font-weight: 700;
        color: #90caf9;
        margin-bottom: 0.2em;
    }
    .subtitle-conference {
        font-size: 1.2rem;
        color: #b0bec5;
        margin-bottom: 1.2em;
    }
    .footer-conference {
        font-size: 0.9rem;
        color: #888;
        margin-top: 2em;
        text-align: center;
    }
    .markdown-text-container, .stMarkdown, .stText, .st-bb, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .st-ca, .st-cb, .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, .st-ck, .st-cl, .st-cm, .st-cn, .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
        color: #e0e0e0 !important;
    }
    .stButton>button {
        background-color: #263238;
        color: #e0e0e0;
        border: 1px solid #90caf9;
    }
    .stButton>button:hover {
        background-color: #90caf9;
        color: #181a1b;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        background-color: #23272b;
        color: #e0e0e0;
        border: 1px solid #90caf9;
    }
    .stSidebar {
        background-color: #23272b !important;
    }
    .stSelectbox>div>div>div>div { color: #90caf9; font-weight: bold; }
    .stMetric {
        background-color: #2e2e2e;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Update menu to include portfolio model selection
menu = st.sidebar.selectbox(
    "Select section:",
    [
        "Introduction",
        "Single Option Analysis",
        "Portfolio Analysis - Black-Scholes",
        "Portfolio Analysis - Binomial",
        "Portfolio Analysis - Monte Carlo"
    ],
    index=0
)

# Personal info card below the sidebar menu
st.sidebar.markdown('''
<div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
    <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
    <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
    <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
</div>
''', unsafe_allow_html=True)

# Implement logic for each portfolio model
if menu == "Portfolio Analysis - Black-Scholes":
    st.write("Black-Scholes portfolio model selected.")
    # Add Black-Scholes portfolio logic here
    S = st.number_input("Spot price (S)", value=100.0, help="Current price of the underlying asset.")
    K = st.number_input("Strike price (K)", value=100.0, help="Strike price of the option.")
    T = st.number_input("Time to maturity (years)", value=1.0, min_value=0.01, help="Time to maturity in years.")
    r = st.number_input("Risk-free rate (r, decimal)", value=0.05, min_value=0.0, max_value=1.0, step=0.01, help="Annual risk-free interest rate.")
    if st.button("Calculate Portfolio", key="bs_portfolio_btn"):
        with st.spinner("Calculating portfolio..."):
            try:
                price, greeks = bsa.calculate_portfolio(S, K, T, r)
                st.metric("Portfolio Price", f"{price:.4f}")
                st.write("Greeks:")
                st.json(greeks)
            except Exception as e:
                st.error(f"Error in calculation: {e}")

elif menu == "Portfolio Analysis - Binomial":
    st.write("Binomial portfolio model selected.")
    # Add Binomial portfolio logic here
    num_options = st.number_input("Number of options in portfolio", min_value=1, max_value=10, value=1, step=1, help="Number of different options in the portfolio.")
    N_steps = st.number_input("Number of steps", value=100, min_value=1, step=1, help="Discretization steps for Binomial model.")
    horizon = st.number_input("Horizon (e.g., enter 10/252 for a 10-day horizon)", value=0.0849, min_value=0.01, format="%.4f", help="Horizon for VaR calculation.")
    portfolio = []
    for i in range(num_options):
        st.subheader(f"Option {i+1}")
        option_style = st.selectbox(f"Option style for Option {i+1}", ["european", "american"], key=f"option_style_{i}", help="Exercise style of the option.")
        option_type = st.selectbox(f"Option type for Option {i+1}", ["call", "put"], key=f"option_type_{i}", help="Call or put option.")
        S = st.number_input(f"Spot price (S) for Option {i+1}", value=5912.17, help="Current price of the underlying asset.", key=f"S_{i}")
        K = st.number_input(f"Strike price (K) for Option {i+1}", value=5915, help="Strike price of the option.", key=f"K_{i}")
        T = st.number_input(f"Time to maturity (years) for Option {i+1}", value=0.0849, min_value=0.01, format="%.4f", help="Time to maturity in years.", key=f"T_{i}")
        r = st.number_input(f"Risk-free rate (r, decimal) for Option {i+1}", value=0.0421, min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", help="Annual risk-free interest rate.", key=f"r_{i}")
        qty = st.number_input(f"Quantity for Option {i+1}", value=-10, step=1, help="Quantity of options in the portfolio.", key=f"qty_{i}")
        market_price = st.number_input(f"Market price for Option {i+1}", value=111.93, help="Observed market price of the option.", key=f"market_price_{i}")
        portfolio.append({'type': option_type, 'style': option_style, 'S': S, 'K': K, 'T': T, 'r': r, 'qty': qty, 'market_price': market_price})
    if st.button("Calculate Portfolio", key="binomial_portfolio_btn"):
        with st.spinner("Calculating portfolio..."):
            try:
                sim_binomial = bpa.simulate_portfolio(portfolio, n_sims=10000, N=N_steps, horizon=T)
                pnl_binomial = sim_binomial['pnl']
                var_binomial, es_binomial = bpa.var_es(pnl_binomial, alpha=0.01)
                value_binomial = bpa.portfolio_value(portfolio, N=N_steps)
                greeks_total_binomial = bpa.portfolio_greeks(portfolio, N=N_steps)
                # Calculate and display model price for each option
                for i, opt in enumerate(portfolio, 1):
                    st.subheader(f"Option {i}")
                    # Calculate model price using existing logic
                    if opt['style'] == 'european':
                        model_price, _ = bpa.price_option(opt, N=N_steps)
                    else:
                        model_price, _ = bpa.price_option(opt, N=N_steps)  # Adjust this line if different logic is needed for American options
                    # Display user-inputted market price
                    market_price = opt['market_price']
                    col1, col2 = st.columns(2)
                    col1.metric("Model Price", f"{model_price:.2f}")
                    col2.metric("Market Price", f"{market_price:.2f}")
                # Add header for risk analysis section
                st.subheader("Risk Metrics")
                # Existing metrics for portfolio analysis
                col1, col2, col3 = st.columns(3)
                col1.metric("Portfolio Value", f"{value_binomial:.2f}")
                col2.metric("VaR (99%)", f"{var_binomial:.2f}")
                col3.metric("ES (99%)", f"{es_binomial:.2f}")
                st.write("Greeks:")
                st.json(greeks_total_binomial)
                # Add histogram for P&L distribution
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.hist(pnl_binomial, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                ax.axvline(-var_binomial, color='red', linestyle='--', label=f'VaR (99%) {-var_binomial:.2f}')
                ax.axvline(-es_binomial, color='orange', linestyle=':', label=f'ES (99%) {-es_binomial:.2f}')
                ax.set_title('Simulated P&L Distribution of the Portfolio (BINOMIAL)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in calculation: {e}")

elif menu == "Portfolio Analysis - Monte Carlo":
    st.write("Monte Carlo portfolio model selected.")
    # Add Monte Carlo portfolio logic here
    num_options = st.number_input("Number of options in portfolio", min_value=1, max_value=10, value=1, step=1, help="Number of different options in the portfolio.")
    n_sim_main = st.number_input("Number of simulations for P&L and VaR/ES", value=50000, min_value=1000, step=1000, help="Number of scenarios for P&L and VaR/ES simulation.")
    n_sim_greeks = st.number_input("Number of simulations for Greeks", value=100000, min_value=1000, step=1000, help="Number of scenarios for Greeks calculation.")
    st.write("Note: The following input uses the Longstaff-Schwartz method.")
    N_steps = st.number_input("Number of steps (For short maturities, use fewer steps; for long maturities, use more steps)", value=100, min_value=1, step=1, help="Discretization steps for Monte Carlo model.")
    horizon = st.number_input("Horizon (e.g., enter 10/252 for a 10-day horizon)", value=0.0849, min_value=0.01, format="%.4f", help="Horizon for VaR calculation.")
    portfolio = []
    for i in range(num_options):
        st.subheader(f"Option {i+1}")
        option_style = st.selectbox(f"Option style for Option {i+1}", ["european", "american"], key=f"option_style_{i}", help="Exercise style of the option.")
        option_type = st.selectbox(f"Option type for Option {i+1}", ["call", "put"], key=f"option_type_{i}", help="Call or put option.")
        S = st.number_input(f"Spot price (S) for Option {i+1}", value=5912.17, help="Current price of the underlying asset.", key=f"S_{i}")
        K = st.number_input(f"Strike price (K) for Option {i+1}", value=5915, help="Strike price of the option.", key=f"K_{i}")
        T = st.number_input(f"Time to maturity (years) for Option {i+1}", value=0.0849, min_value=0.01, format="%.4f", help="Time to maturity in years.", key=f"T_{i}")
        r = st.number_input(f"Risk-free rate (r, decimal) for Option {i+1}", value=0.0421, min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", help="Annual risk-free interest rate.", key=f"r_{i}")
        qty = st.number_input(f"Quantity for Option {i+1}", value=-10, step=1, help="Quantity of options in the portfolio.", key=f"qty_{i}")
        market_price = st.number_input(f"Market price for Option {i+1}", value=111.93, help="Observed market price of the option.", key=f"market_price_{i}")
        portfolio.append({'type': option_type, 'style': option_style, 'S': S, 'K': K, 'T': T, 'r': r, 'qty': qty, 'market_price': market_price})
    if st.button("Calculate Portfolio", key="mc_portfolio_btn"):
        with st.spinner("Calculating portfolio..."):
            try:
                sim_mc = mca.simulate_portfolio_mc_pricing(portfolio, n_sims=n_sim_main, n_steps=N_steps, horizon=horizon)
                pnl_mc = sim_mc['pnl']
                var_mc, es_mc = mca.var_es(pnl_mc, alpha=0.01)
                value_mc = sum(mca.price_option_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps) * opt['qty'] for opt in portfolio)
                greeks_total_mc = mca.portfolio_greeks_mc(portfolio, n_sim=n_sim_greeks, n_steps=N_steps)
                # Calculate and display model price for each option
                for i, opt in enumerate(portfolio, 1):
                    st.subheader(f"Option {i}")
                    # Calculate model price using existing logic
                    model_price = mca.price_option_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps)
                    # Display user-inputted market price
                    market_price = opt['market_price']
                    col1, col2 = st.columns(2)
                    col1.metric("Model Price", f"{model_price:.2f}")
                    col2.metric("Market Price", f"{market_price:.2f}")
                # Add header for risk analysis section
                st.subheader("Risk Metrics")
                # Existing metrics for portfolio analysis
                col1, col2, col3 = st.columns(3)
                col1.metric("Portfolio Value", f"{value_mc:.2f}")
                col2.metric("VaR (99%)", f"{var_mc:.2f}")
                col3.metric("ES (99%)", f"{es_mc:.2f}")
                st.write("Greeks:")
                st.json(greeks_total_mc)
                # Add histogram for P&L distribution
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.hist(pnl_mc, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                ax.axvline(-var_mc, color='red', linestyle='--', label=f'VaR (99%) {-var_mc:.2f}')
                ax.axvline(-es_mc, color='orange', linestyle=':', label=f'ES (99%) {-es_mc:.2f}')
                ax.set_title('Simulated P&L Distribution of the Portfolio (MONTE CARLO)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in calculation: {e}")

if menu == "Introduction":
    st.markdown('<div class="title-conference">Option Pricing & Portfolio Risk App</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-conference">A robust and visual tool for pricing European and American options, portfolio risk analysis, and model comparison.</div>', unsafe_allow_html=True)
    st.markdown("""
    **Instructions:**
    - Select a section from the sidebar.
    - For single options, choose model and type (European/American).
    - For portfolios, define your positions and analyze risk.
    - View and compare results visually.
    """)
    st.markdown("""
    **About:**
    This application is designed for quantitative finance professionals, students, and researchers. It provides a unified interface to explore and compare the most important models for option pricing and portfolio risk, including Greeks, VaR, ES, and more.
    """)

if menu == "Single Option Analysis":
    st.header("Single Option Analysis")
    cols = st.columns(3)
    with cols[0]:
        model = st.selectbox("Model", ["Black-Scholes", "Binomial", "Monte Carlo"], key="model_single", help="Pricing model for this option.")
        option_style = st.selectbox("Option style", ["European", "American"], key="option_style_single", help="Exercise style of the option.")
        option_type = st.selectbox("Option type", ["call", "put"], key="option_type_single", help="Call or put option.")
    with cols[1]:
        S = st.number_input("Spot price (S)", value=st.session_state.get("S_single", 100.0), key="S_single", help="Current price of the underlying asset.")
        K = st.number_input("Strike price (K)", value=st.session_state.get("K_single", 100.0), key="K_single", help="Strike price of the option.")
        T = st.number_input("Time to maturity (years)", value=st.session_state.get("T_single", 1.0), key="T_single", min_value=0.01, help="Time to maturity in years.")
    with cols[2]:
        r = st.number_input("Risk-free rate (r, decimal)", value=st.session_state.get("r_single", 0.05), key="r_single", min_value=0.0, max_value=1.0, step=0.01, help="Annual risk-free interest rate (as decimal, e.g. 0.03 for 3%).")
        market_price = st.number_input("Option market price", value=st.session_state.get("market_price_single", 10.0), key="market_price_single", min_value=0.0, help="Observed market price of the option.")
        N = st.number_input("Number of steps (Binomial/MC)", value=st.session_state.get("N_single", 100), min_value=1, step=1, key="N_single", help="Discretization steps for Binomial/Monte Carlo models.")
        n_sim = st.number_input("Number of Monte Carlo simulations", value=st.session_state.get("n_sim_single", 10000), min_value=1000, step=1000, key="n_sim_single", help="Number of scenarios for risk simulation.")
    # Validación básica
    if T <= 0:
        st.error("Maturity must be positive.")
    if st.button("Calculate Option", key="single_option_btn"):
        with st.spinner("Calculating option price and Greeks..."):
            try:
                if model == "Black-Scholes":
                    if option_style == "European":
                        opt = {'type': option_type, 'S': S, 'K': K, 'T': T, 'r': r, 'market_price': market_price}
                        price, iv = bsa.price_option(opt)
                        greeks = bsa.option_greeks(opt)
                    else:
                        st.info("Black-Scholes is not suitable for American options. Use Binomial or Monte Carlo.")
                        price = None
                        greeks = None
                elif model == "Binomial":
                    if option_style == "European":
                        iv = bpa.implied_volatility_option(market_price, S, K, T, r, option_type)
                        price = bpa.binomial_european_option_price(S, K, T, r, iv, int(N), option_type)
                        greeks = bpa.binomial_greeks_european_option(S, K, T, r, iv, int(N), option_type)
                    else:
                        iv = bsa.implied_volatility_option(market_price, S, K, T, r, option_type)  # Use Black-Scholes for IV
                        price = bpa.binomial_american_option_price(S, K, T, r, iv, int(N), option_type)
                        greeks = bpa.binomial_greeks_american_option(S, K, T, r, iv, int(N), option_type)
                elif model == "Monte Carlo":
                    opt = {'type': option_type, 'style': option_style.lower(), 'S': S, 'K': K, 'T': T, 'r': r, 'qty': 1, 'market_price': market_price}
                    price = mca.price_option_mc(opt, n_sim=int(n_sim), n_steps=int(N))
                    greeks = mca.option_greeks_mc(opt, n_sim=int(n_sim), n_steps=int(N))
                    iv = mca.implied_volatility_option(market_price, S, K, T, r, option_type)
                # Resultados destacados
                col1, col2 = st.columns(2)
                if price is not None:
                    col1.metric("Model Price", f"{price:.4f}")
                if iv is not None:
                    col2.metric("Implied Volatility", f"{iv:.4f}")
                st.markdown("---")
                if greeks is not None:
                    st.write("Greeks:")
                    st.json(greeks)
            except Exception as e:
                st.error(f"Error in calculation: {e}")

st.markdown('<div class="footer-conference">Developed by Marcos Heredia Pimienta, Quantitative Risk Analyst</div>', unsafe_allow_html=True)
