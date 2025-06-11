import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from option_pricing.black_scholes_model import bs_portfolio_analysis as bsa
from option_pricing.binomial_model import binomial_portfolio_analysis as bpa
from option_pricing.monte_carlo import mc_portfolio_analysis as mca
from portfolio_management.scripts.fundamental_analysis import FinancialAnalyzer
from portfolio_management.scripts.technical_analysis import (
    descargar_datos, calcular_sma_multiple, calcular_ema_multiple, calcular_rsi,
    calcular_macd, calcular_bollinger_bands, calcular_momentum, calcular_adx,
    calcular_obv, calcular_stochastic_oscillator, plot_candlestick_and_momentum,
    plot_candlestick_and_rsi, plot_candlestick_and_macd, plot_candlestick_and_bollinger,
    plot_sma_multiple, plot_ema_multiple, plot_adx, plot_stochastic_oscillator,
    plot_macd_with_adx, plot_macd_with_stochastic, plot_rsi_with_adx, plot_rsi_with_stochastic)

import glob

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

# Update menu to include Fundamental Analysis in the first selectbox
menu1 = st.sidebar.selectbox(
    "Select section:",
    [
        "Introduction",
        "Single Option Analysis",
        "Portfolio Analysis - Black-Scholes",
        "Portfolio Analysis - Binomial",
        "Portfolio Analysis - Monte Carlo",
        "Hedging Strategy",
        "Sensitivity Analysis",
        "Fundamental Analysis",
        "Technical Analysis"
    ],
    index=0
)

def format_number(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}".replace(',', '.').replace('.', ',', 1)

if menu1 == "Introduction":
    st.markdown('<div class="title-conference">Option Pricing & Portfolio Risk App</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-conference">A cutting-edge tool for pricing options, analyzing portfolio risk, and comparing financial models.</div>', unsafe_allow_html=True)
    st.markdown("""
    **🚀 Quick Start Guide:**
    - **Navigate** through the sections using the sidebar.
    - For single options, **select your model** and option type (European/American).
    - For portfolios, **define your positions** and assess risk metrics.
    - **Visualize** and compare results with interactive charts.
    """)
    st.markdown("""
    **💡 About This App:**
    Designed for quantitative finance professionals, students, and researchers, this application offers a comprehensive interface to explore and compare key models for option pricing and portfolio risk. All within a user-friendly environment.
    
    **Key Features:**
    - 📊 **Interactive Visualizations**: Gain insights with dynamic charts and graphs.
    - 📈 **Comprehensive Analysis**: Dive into detailed analyses of Greeks, VaR, ES, and more.
    - 🔍 **Model Comparison**: Evaluate different financial models side by side.
    - 🛠️ **User-Friendly Interface**: Intuitive design for seamless navigation.
    
    **Get Started Now!**
    """)

    # Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if menu1 == "Single Option Analysis":
    st.header("🔍 Single Option Analysis")
    st.write("Select your model and option type to begin analyzing single options")
    cols = st.columns(3)
    with cols[0]:
        model = st.selectbox("Model", ["Black-Scholes", "Binomial", "Monte Carlo"], key="model_single", help="Choose the pricing model for this option.")
        option_style = st.selectbox("Option style", ["European", "American"], key="option_style_single", help="Select the exercise style of the option.")
        option_type = st.selectbox("Option type", ["call", "put"], key="option_type_single", help="Choose call or put option.")
    with cols[1]:
        S = st.number_input("Spot price (S)", value=st.session_state.get("S_single", 100.0), key="S_single", help="Enter the current price of the underlying asset.")
        K = st.number_input("Strike price (K)", value=st.session_state.get("K_single", 100.0), key="K_single", help="Enter the strike price of the option.")
        T = st.number_input("Time to maturity (years)", value=st.session_state.get("T_single", 1.0), key="T_single", min_value=0.01, help="Specify the time to maturity in years.")
    with cols[2]:
        r = st.number_input("Risk-free rate (r, decimal)", value=st.session_state.get("r_single", 0.05), key="r_single", min_value=0.0, max_value=1.0, step=0.01, help="Input the annual risk-free interest rate.")
        market_price = st.number_input("Option market price", value=st.session_state.get("market_price_single", 10.0), key="market_price_single", min_value=0.0, help="Provide the observed market price of the option.")
        if model == "Binomial":
            N_steps = st.number_input("Number of steps (Binomial)", value=st.session_state.get("N_single", 100), min_value=1, step=1, key="N_single", help="Set the discretization steps for the Binomial model.")
        if model == "Monte Carlo":
            n_sim = st.number_input("Number of Monte Carlo simulations", value=st.session_state.get("n_sim_single", 10000), min_value=1000, step=1000, key="n_sim_single", help="Define the number of scenarios for risk simulation.")
            N_steps = 1  # Default to 1 for European options
            if option_style == "American":
                N_steps = st.number_input("Number of steps (Monte Carlo)", value=st.session_state.get("N_steps_single", 100), min_value=1, step=1, key="N_steps_single", help="Set the discretization steps for the Monte Carlo model.")
    # Basic validation
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
                        price = bpa.binomial_european_option_price(S, K, T, r, iv, int(N_steps), option_type)
                        greeks = bpa.binomial_greeks_european_option(S, K, T, r, iv, int(N_steps), option_type)
                    else:
                        iv = bsa.implied_volatility_option(market_price, S, K, T, r, option_type)  # Use Black-Scholes for IV
                        price = bpa.binomial_american_option_price(S, K, T, r, iv, int(N_steps), option_type)
                        greeks = bpa.binomial_greeks_american_option(S, K, T, r, iv, int(N_steps), option_type)
                elif model == "Monte Carlo":
                    opt = {'type': option_type, 'style': option_style.lower(), 'S': S, 'K': K, 'T': T, 'r': r, 'qty': 1, 'market_price': market_price}
                    price = mca.price_option_mc(opt, n_sim=int(n_sim), n_steps=int(N_steps))
                    greeks = mca.option_greeks_mc(opt, n_sim=int(n_sim), n_steps=int(N_steps))
                    iv = mca.implied_volatility_option(market_price, S, K, T, r, option_type)
                # Highlighted results
                col1, col2 = st.columns(2)
                if price is not None:
                    col1.metric("Model Price", f"{price:.2f}")
                if iv is not None:
                    col2.metric("Implied Volatility", f"{iv:.2f}")
                st.markdown("---")
                if greeks is not None:
                    st.write("Greeks:")
                    st.json(greeks)
            except Exception as e:
                st.error(f"Error in calculation: {e}")
     # Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if menu1 == "Portfolio Analysis - Black-Scholes":
    st.header("📊 Portfolio Analysis - Black-Scholes")
    st.subheader("Analyze your portfolio using the Black-Scholes model")
    st.write("Configure your portfolio and analyze risk metrics with precision")
    num_options = st.number_input("Number of options in portfolio", min_value=1, max_value=10, value=4, step=1, help="Specify the number of different options in the portfolio.")
    horizon = st.number_input("Horizon (e.g., enter 10/252 for a 10-day horizon)", value=0.0849, min_value=0.01, format="%.4f", help="Set the horizon for VaR calculation.")
    portfolio = []
    default_values = [
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5915, 'T': 0.0849, 'r': 0.0421, 'qty': -10, 'market_price': 111.93},
        {'type': 'put', 'style': 'european', 'S': 5912.17, 'K': 5910, 'T': 0.0849, 'r': 0.0421, 'qty': -5, 'market_price': 106.89},
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5920, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 103.66},
        {'type': 'put', 'style': 'european', 'S': 5912.17, 'K': 5900, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 102.92}
    ]
    for i in range(num_options):
        st.subheader(f"Option {i+1}")
        option_type = st.selectbox(f"Option type for Option {i+1}", ["call", "put"], index=["call", "put"].index(default_values[i]['type']), key=f"option_type_{i}", help="Select call or put option.")
        option_style = st.selectbox(f"Option style for Option {i+1}", ["european", "american"], index=["european", "american"].index(default_values[i]['style']), key=f"option_style_{i}", help="Choose the exercise style of the option.")
        S = st.number_input(f"Spot price (S) for Option {i+1}", value=default_values[i]['S'], help="Enter the current price of the underlying asset.", key=f"S_{i}")
        K = st.number_input(f"Strike price (K) for Option {i+1}", value=default_values[i]['K'], help="Enter the strike price of the option.", key=f"K_{i}")
        T = st.number_input(f"Time to maturity (years) for Option {i+1}", value=default_values[i]['T'], min_value=0.01, format="%.4f", help="Specify the time to maturity in years.", key=f"T_{i}")
        r = st.number_input(f"Risk-free rate (r, decimal) for Option {i+1}", value=default_values[i]['r'], min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", help="Input the annual risk-free interest rate.", key=f"r_{i}")
        qty = st.number_input(f"Quantity for Option {i+1}", value=default_values[i]['qty'], step=1, help="Specify the quantity of options in the portfolio.", key=f"qty_{i}")
        market_price = st.number_input(f"Market price for Option {i+1}", value=default_values[i]['market_price'], help="Provide the observed market price of the option.", key=f"market_price_{i}")
        portfolio.append({'type': option_type, 'style': option_style, 'S': S, 'K': K, 'T': T, 'r': r, 'qty': qty, 'market_price': market_price})
    if st.button("Calculate Portfolio", key="bs_portfolio_btn"):
        with st.spinner("Calculating portfolio..."):
            try:
                np.random.seed(42)  # Set seed for reproducibility
                value_bs = bsa.portfolio_value(portfolio)
                greeks_total_bs = bsa.portfolio_greeks(portfolio)
                # Simulate P&L distribution using bs_portfolio_analysis
                sim_bs = bsa.simulate_portfolio(portfolio, n_sims=10000, horizon=horizon)
                pnl_bs = sim_bs['pnl']
                # Calculate var_bs before using it
                var_bs, es_bs = bsa.var_es(pnl_bs, alpha=0.01)
                # Display portfolio Greeks first
                st.write("Portfolio Greeks:")
                st.json(greeks_total_bs)
                # Display individual option model vs market prices
                for i, opt in enumerate(portfolio, 1):
                    st.subheader(f"Option {i} Details")
                    model_price, _ = bsa.price_option(opt)
                    col1, col2 = st.columns(2)
                    col1.metric("Model Price", f"{model_price:.2f}")
                    col2.metric("Market Price", f"{opt['market_price']:.2f}")
                # Add Metric Risk label
                st.subheader("Metric Risk")
                # Display portfolio value, VaR, and ES horizontally
                col1, col2, col3 = st.columns(3)
                col1.metric("Portfolio Value", f"{value_bs:.2f}")
                col2.metric("VaR (99%)", f"{var_bs:.2f}")
                col3.metric("ES (99%)", f"{es_bs:.2f}")
                # Plot histogram of P&L distribution
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.hist(pnl_bs, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True)
                ax.axvline(-var_bs, color='red', linestyle='--', label=f'VaR (99%) {-var_bs:.2f}')
                ax.axvline(-es_bs, color='orange', linestyle=':', label=f'ES (99%) {-es_bs:.2f}')
                ax.set_title('Simulated P&L Distribution of the Portfolio (Black-Scholes)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in calculation: {e}")
     # Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

elif menu1 == "Portfolio Analysis - Binomial":
    st.header("📊 Portfolio Analysis - Binomial")
    st.subheader("Analyze your portfolio using the Binomial model")
    st.write("Configure your portfolio and assess risk metrics with the Binomial model")
    num_options = st.number_input("Number of options in portfolio", min_value=1, max_value=10, value=4, step=1, help="Specify the number of different options in the portfolio.")
    N_steps = st.number_input("Number of steps", value=100, min_value=1, step=1, help="Set the discretization steps for the Binomial model.")
    horizon = st.number_input("Horizon (e.g., enter 10/252 for a 10-day horizon)", value=0.0849, min_value=0.01, format="%.4f", help="Set the horizon for VaR calculation.")
    portfolio = []
    default_values = [
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5915, 'T': 0.0849, 'r': 0.0421, 'qty': -10, 'market_price': 111.93},
        {'type': 'put', 'style': 'american', 'S': 5912.17, 'K': 5910, 'T': 0.0849, 'r': 0.0421, 'qty': -5, 'market_price': 106.89},
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5920, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 103.66},
        {'type': 'put', 'style': 'european', 'S': 5912.17, 'K': 5900, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 102.92}
    ]
    for i in range(num_options):
        st.subheader(f"Option {i+1}")
        option_style = st.selectbox(f"Option style for Option {i+1}", ["european", "american"], index=["european", "american"].index(default_values[i]['style']), key=f"option_style_{i}", help="Choose the exercise style of the option.")
        option_type = st.selectbox(f"Option type for Option {i+1}", ["call", "put"], index=["call", "put"].index(default_values[i]['type']), key=f"option_type_{i}", help="Select call or put option.")
        S = st.number_input(f"Spot price (S) for Option {i+1}", value=default_values[i]['S'], help="Enter the current price of the underlying asset.", key=f"S_{i}")
        K = st.number_input(f"Strike price (K) for Option {i+1}", value=default_values[i]['K'], help="Enter the strike price of the option.", key=f"K_{i}")
        T = st.number_input(f"Time to maturity (years) for Option {i+1}", value=default_values[i]['T'], min_value=0.01, format="%.4f", help="Specify the time to maturity in years.", key=f"T_{i}")
        r = st.number_input(f"Risk-free rate (r, decimal) for Option {i+1}", value=default_values[i]['r'], min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", help="Input the annual risk-free interest rate.", key=f"r_{i}")
        qty = st.number_input(f"Quantity for Option {i+1}", value=default_values[i]['qty'], step=1, help="Specify the quantity of options in the portfolio.", key=f"qty_{i}")
        market_price = st.number_input(f"Market price for Option {i+1}", value=default_values[i]['market_price'], help="Provide the observed market price of the option.", key=f"market_price_{i}")
        portfolio.append({'type': option_type, 'style': option_style, 'S': S, 'K': K, 'T': T, 'r': r, 'qty': qty, 'market_price': market_price})
    if st.button("Calculate Portfolio", key="binomial_portfolio_btn"):
        with st.spinner("Calculating portfolio..."):
            try:
                sim_binomial = bpa.simulate_portfolio(portfolio, n_sims=10000, N=N_steps, horizon=horizon)
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
     # Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

elif menu1 == "Portfolio Analysis - Monte Carlo":
    st.header("📊 Portfolio Analysis - Monte Carlo")
    st.subheader("Analyze your portfolio using the Monte Carlo model")
    st.write("Configure your portfolio and evaluate risk metrics with the Monte Carlo simulation.")
    num_options = st.number_input("Number of options in portfolio", min_value=1, max_value=10, value=4, step=1, help="Specify the number of different options in the portfolio.")
    n_sim_main = st.number_input("Number of simulations for P&L and VaR/ES", value=50000, min_value=1000, step=1000, help="Define the number of scenarios for P&L and VaR/ES simulation.")
    n_sim_greeks = st.number_input("Number of simulations for Greeks", value=100000, min_value=1000, step=1000, help="Specify the number of scenarios for Greeks calculation.")
    st.write("Note: The following input uses the Longstaff-Schwartz method.")
    N_steps = st.number_input("Number of steps ", value=st.session_state.get("N_steps_single", 100), min_value=1, step=1, key="N_steps_single", help="Set the discretization steps for the Monte Carlo model.")
    horizon = st.number_input("Horizon (e.g., enter 10/252 for a 10-day horizon)", value=0.0849, min_value=0.01, format="%.4f", help="Set the horizon for VaR calculation.")
    portfolio = []
    default_values = [
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5915, 'T': 0.0849, 'r': 0.0421, 'qty': -10, 'market_price': 111.93},
        {'type': 'put', 'style': 'american', 'S': 5912.17, 'K': 5910, 'T': 0.0849, 'r': 0.0421, 'qty': -5, 'market_price': 106.89},
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5920, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 103.66},
        {'type': 'put', 'style': 'european', 'S': 5912.17, 'K': 5900, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 102.92}
    ]
    for i in range(num_options):
        st.subheader(f"Option {i+1}")
        option_style = st.selectbox(f"Option style for Option {i+1}", ["european", "american"], index=["european", "american"].index(default_values[i]['style']), key=f"option_style_{i}", help="Choose the exercise style of the option.")
        option_type = st.selectbox(f"Option type for Option {i+1}", ["call", "put"], index=["call", "put"].index(default_values[i]['type']), key=f"option_type_{i}", help="Select call or put option.")
        S = st.number_input(f"Spot price (S) for Option {i+1}", value=default_values[i]['S'], help="Enter the current price of the underlying asset.", key=f"S_{i}")
        K = st.number_input(f"Strike price (K) for Option {i+1}", value=default_values[i]['K'], help="Enter the strike price of the option.", key=f"K_{i}")
        T = st.number_input(f"Time to maturity (years) for Option {i+1}", value=default_values[i]['T'], min_value=0.01, format="%.4f", help="Specify the time to maturity in years.", key=f"T_{i}")
        r = st.number_input(f"Risk-free rate (r, decimal) for Option {i+1}", value=default_values[i]['r'], min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", help="Input the annual risk-free interest rate.", key=f"r_{i}")
        qty = st.number_input(f"Quantity for Option {i+1}", value=default_values[i]['qty'], step=1, help="Specify the quantity of options in the portfolio.", key=f"qty_{i}")
        market_price = st.number_input(f"Market price for Option {i+1}", value=default_values[i]['market_price'], help="Provide the observed market price of the option.", key=f"market_price_{i}")
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
                ax.set_title('Simulated P&L Distribution of the Portfolio (Monte Carlo)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in calculation: {e}")
     # Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if menu1 == "Hedging Strategy":
    st.header("🛡️ Hedging Strategy")
    st.subheader("Optimize and protect your portfolio by choosing the right hedging")
    st.write("Select your model and hedging strategy to manage portfolio risk effectively")
    model = st.selectbox("Select Model:", ["Black-Scholes", "Binomial", "Monte Carlo"], index=0)
    hedge_strategy = st.selectbox("Select Hedging Strategy:", ["Delta Hedge", "Delta+Gamma Hedge", "Vega Hedge"], index=0)
    coverage_percentage = st.number_input("Hedging (%):", value=70, min_value=0, max_value=100, step=1, help="Specify the percentage of the portfolio to be hedged.")
    horizon = st.number_input("Horizon (e.g., enter 10/252 for a 10-day horizon)", value=0.0849, min_value=0.01, format="%.4f", help="Set the horizon for VaR calculation.")
    num_options = st.number_input("Number of options in portfolio", min_value=1, max_value=10, value=4, step=1, help="Specify the number of different options in the portfolio.")
    portfolio = []
    default_values = [
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5915, 'T': 0.0849, 'r': 0.0421, 'qty': -10, 'market_price': 111.93},
        {'type': 'put', 'style': 'american', 'S': 5912.17, 'K': 5910, 'T': 0.0849, 'r': 0.0421, 'qty': -5, 'market_price': 106.89},
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5920, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 103.66},
        {'type': 'put', 'style': 'european', 'S': 5912.17, 'K': 5900, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 102.92}
    ]
    if model == "Monte Carlo":
        n_sim_main = st.number_input("Number of simulations for P&L and VaR/ES", value=50000, min_value=1000, step=1000, help="Define the number of scenarios for P&L and VaR/ES simulation.")
        n_sim_greeks = st.number_input("Number of simulations for Greeks", value=100000, min_value=1000, step=1000, help="Specify the number of scenarios for Greeks calculation.")
        st.write("Note: The following input uses the Longstaff-Schwartz method.")
        N_steps = st.number_input("Number of steps (For short maturities, use fewer steps; for long maturities, use more steps)", value=100, min_value=1, step=1, help="Set the discretization steps for the Monte Carlo model.")
    elif model == "Black-Scholes":
        n_sim_main = st.number_input("Number of simulations for P&L and VaR/ES", value=50000, min_value=1000, step=1000, help="Define the number of scenarios for P&L and VaR/ES simulation.")
    elif model == "Binomial":
        n_sim_main = st.number_input("Number of simulations for P&L and VaR/ES", value=50000, min_value=1000, step=1000, help="Define the number of scenarios for P&L and VaR/ES simulation.")
        N_steps_binomial = st.number_input("Number of steps for Binomial tree", value=100, min_value=1, step=1, help="Set the discretization steps for the Binomial model.")
    for i in range(num_options):
        st.subheader(f"Option {i+1}")
        option_type = st.selectbox(f"Option type for Option {i+1}", ["call", "put"], index=["call", "put"].index(default_values[i]['type']), key=f"option_type_{i}", help="Select call or put option.")
        option_style = st.selectbox(f"Option style for Option {i+1}", ["european", "american"], index=["european", "american"].index(default_values[i]['style']), key=f"option_style_{i}", help="Choose the exercise style of the option.")
        S = st.number_input(f"Spot price (S) for Option {i+1}", value=default_values[i]['S'], help="Enter the current price of the underlying asset.", key=f"S_{i}")
        K = st.number_input(f"Strike price (K) for Option {i+1}", value=default_values[i]['K'], help="Enter the strike price of the option.", key=f"K_{i}")
        T = st.number_input(f"Time to maturity (years) for Option {i+1}", value=default_values[i]['T'], min_value=0.01, format="%.4f", help="Specify the time to maturity in years.", key=f"T_{i}")
        r = st.number_input(f"Risk-free rate (r, decimal) for Option {i+1}", value=default_values[i]['r'], min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", help="Input the annual risk-free interest rate.", key=f"r_{i}")
        qty = st.number_input(f"Quantity for Option {i+1}", value=default_values[i]['qty'], step=1, help="Specify the quantity of options in the portfolio.", key=f"qty_{i}")
        market_price = st.number_input(f"Market price for Option {i+1}", value=default_values[i]['market_price'], help="Provide the observed market price of the option.", key=f"market_price_{i}")
        portfolio.append({'type': option_type, 'style': option_style, 'S': S, 'K': K, 'T': T, 'r': r, 'qty': qty, 'market_price': market_price})
    if st.button("Calculate Hedging Strategy", key="hedging_strategy_btn"):
        with st.spinner("Calculating hedging strategy..."):
            try:
                if model == "Black-Scholes":
                    value_bs = bsa.portfolio_value(portfolio)
                    sim_bs = bsa.simulate_portfolio(portfolio, n_sims=10000, horizon=horizon)
                    pnl_bs = sim_bs['pnl']
                    shocks_bs = sim_bs['shocks']
                    var_bs, es_bs = bsa.var_es(pnl_bs, alpha=0.01)
                    if hedge_strategy == "Delta Hedge":
                        delta_hedge_fraction_bs = coverage_percentage / 100.0
                        subyacentes = {}
                        for opt in portfolio:
                            key = opt.get('ticker', opt['S'])
                            greeks = bsa.option_greeks(opt)
                            subyacentes.setdefault(key, {'S0': opt['S'], 'delta': 0})
                            subyacentes[key]['delta'] += greeks['delta'] * opt['qty'] * delta_hedge_fraction_bs
                        pnl_bs_hedged = []
                        for i in range(len(pnl_bs)):
                            hedge_pnl = 0
                            for key, v in subyacentes.items():
                                S0 = v['S0']
                                delta = v['delta']
                                Z = shocks_bs[key][i]
                                for opt in portfolio:
                                    if opt.get('ticker', opt['S']) == key:
                                        iv = bsa.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                        if iv is None:
                                            iv = 0.2
                                        r = opt['r']
                                        T_sim = horizon if horizon is not None else opt['T']
                                        break
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                hedge_pnl += -delta * (S_T - S0)
                            pnl_bs_hedged.append(pnl_bs[i] + hedge_pnl)
                        pnl_bs_hedged = np.array(pnl_bs_hedged)
                        var_bs_hedged, es_bs_hedged = bsa.var_es(pnl_bs_hedged, alpha=0.01)
                        st.write(f"Delta hedge per underlying:")
                        for key, v in subyacentes.items():
                            st.write(f"  Underlying {key}: delta = {v['delta']:.4f}")
                        st.write(f"\nVaR after delta hedge (BS, 99%): {var_bs_hedged:.2f}")
                        st.write(f"ES after delta hedge (BS, 99%): {es_bs_hedged:.2f}")
                        st.write(f"VaR reduction: {var_bs - var_bs_hedged:.2f}")
                        st.write(f"ES reduction: {es_bs - es_bs_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_bs, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_bs_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
                        ax.axvline(-var_bs, color='blue', linestyle='--', label=f'VaR Original ({-var_bs:.2f})')
                        ax.axvline(-es_bs, color='blue', linestyle=':', label=f'ES Original ({-es_bs:.2f})')
                        ax.axvline(-var_bs_hedged, color='red', linestyle='--', label=f'VaR Delta ({-var_bs_hedged:.2f})')
                        ax.axvline(-es_bs_hedged, color='red', linestyle=':', label=f'ES Delta ({-es_bs_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Black-Scholes)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
                    elif hedge_strategy == "Delta+Gamma Hedge":
                        greeks_total = bsa.portfolio_greeks(portfolio)
                        gamma_cartera = greeks_total['gamma']
                        delta_cartera = greeks_total['delta']
                        hedge_opt = {
                            'type': 'call',
                            'style': 'european',
                            'S': portfolio[0]['S'],
                            'K': portfolio[0]['K'],
                            'T': portfolio[0]['T'],
                            'r': portfolio[0]['r'],
                            'qty': 0,
                            'market_price': portfolio[0]['market_price'],
                        }
                        greeks_hedge = bsa.option_greeks(hedge_opt)
                        gamma_hedge = greeks_hedge['gamma']
                        gamma_hedge_fraction = coverage_percentage / 100.0
                        qty_gamma_hedge = -gamma_cartera * gamma_hedge_fraction / gamma_hedge if gamma_hedge != 0 else 0
                        hedge_opt['qty'] = qty_gamma_hedge
                        portfolio_gamma_hedged = portfolio + [hedge_opt]
                        greeks_total_gamma = bsa.portfolio_greeks(portfolio_gamma_hedged)
                        delta_gamma_hedged = greeks_total_gamma['delta']
                        pnl_gamma_delta_hedged = []
                        for i in range(len(pnl_bs)):
                            shocked_portfolio = []
                            for opt in portfolio_gamma_hedged:
                                key = opt.get('ticker', opt['S'])
                                S0 = opt['S']
                                iv = bsa.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                if iv is None:
                                    iv = 0.2
                                r = opt['r']
                                T_sim = horizon if horizon is not None else opt['T']
                                Z = shocks_bs[key][i] if key in shocks_bs else np.random.normal(0, 1)
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                shocked_opt = opt.copy()
                                shocked_opt['S'] = S_T
                                shocked_portfolio.append(shocked_opt)
                            val = bsa.portfolio_value(shocked_portfolio)
                            hedge_pnl = 0
                            S0 = portfolio[0]['S']
                            delta = delta_gamma_hedged
                            Z = shocks_bs[portfolio[0].get('ticker', portfolio[0]['S'])][i]
                            S_T = S0 * np.exp((portfolio[0]['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                            hedge_pnl += -delta * (S_T - S0)
                            pnl_gamma_delta_hedged.append(val - value_bs + hedge_pnl)
                        pnl_gamma_delta_hedged = np.array(pnl_gamma_delta_hedged)
                        var_gamma_delta_hedged, es_gamma_delta_hedged = bsa.var_es(pnl_gamma_delta_hedged, alpha=0.01)
                        st.write(f"Gamma hedge: qty = {qty_gamma_hedge:.4f} of ATM call (S={hedge_opt['S']}, K={hedge_opt['K']}) covering {gamma_hedge_fraction*100:.0f}% of gamma")
                        st.write(f"Delta after gamma hedge: {delta_gamma_hedged:.4f}")
                        st.write(f"\nVaR after gamma+delta hedge (BS, 99%): {var_gamma_delta_hedged:.2f}")
                        st.write(f"ES after gamma+delta hedge (BS, 99%): {es_gamma_delta_hedged:.2f}")
                        st.write(f"VaR reduction vs original: {var_bs - var_gamma_delta_hedged:.2f}")
                        st.write(f"ES reduction vs original: {es_bs - es_gamma_delta_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_bs, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_gamma_delta_hedged, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
                        ax.axvline(-var_bs, color='blue', linestyle='--', label=f'VaR Original ({-var_bs:.2f})')
                        ax.axvline(-es_bs, color='blue', linestyle=':', label=f'ES Original ({-es_bs:.2f})')
                        ax.axvline(-var_gamma_delta_hedged, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_gamma_delta_hedged:.2f})')
                        ax.axvline(-es_gamma_delta_hedged, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_gamma_delta_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Black-Scholes)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
                    elif hedge_strategy == "Vega Hedge":
                        vega_total = bsa.portfolio_greeks(portfolio)['vega']
                        hedge_opt_vega = {
                            'type': 'call',
                            'style': 'european',
                            'S': portfolio[0]['S'],
                            'K': portfolio[0]['K'],
                            'T': portfolio[0]['T'],
                            'r': portfolio[0]['r'],
                            'qty': 0,
                            'market_price': portfolio[0]['market_price'],
                        }
                        greeks_hedge_vega = bsa.option_greeks(hedge_opt_vega)
                        vega_hedge = greeks_hedge_vega['vega']
                        vega_hedge_fraction = coverage_percentage / 100.0
                        qty_vega_hedge = -vega_total * vega_hedge_fraction / vega_hedge if vega_hedge != 0 else 0
                        hedge_opt_vega['qty'] = qty_vega_hedge
                        portfolio_vega_hedged = portfolio + [hedge_opt_vega]
                        pnl_vega_hedged = []
                        for i in range(len(pnl_bs)):
                            shocked_portfolio = []
                            for opt in portfolio_vega_hedged:
                                key = opt.get('ticker', opt['S'])
                                S0 = opt['S']
                                iv = bsa.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                if iv is None:
                                    iv = 0.2
                                r = opt['r']
                                T_sim = horizon if horizon is not None else opt['T']
                                Z = shocks_bs[key][i] if key in shocks_bs else np.random.normal(0, 1)
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                shocked_opt = opt.copy()
                                shocked_opt['S'] = S_T
                                shocked_portfolio.append(shocked_opt)
                            val = bsa.portfolio_value(shocked_portfolio)
                            pnl_vega_hedged.append(val - value_bs)
                        pnl_vega_hedged = np.array(pnl_vega_hedged)
                        var_vega_hedged, es_vega_hedged = bsa.var_es(pnl_vega_hedged, alpha=0.01)
                        st.write(f"Vega hedge: qty = {qty_vega_hedge:.4f} of ATM call (S={hedge_opt_vega['S']}, K={hedge_opt_vega['K']}) covering {vega_hedge_fraction*100:.0f}% of vega")
                        st.write(f"\nVaR after vega hedge (BS, 99%): {var_vega_hedged:.2f}")
                        st.write(f"ES after vega hedge (BS, 99%): {es_vega_hedged:.2f}")
                        st.write(f"VaR reduction vs original: {var_bs - var_vega_hedged:.2f}")
                        st.write(f"ES reduction vs original: {es_bs - es_vega_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_bs, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_vega_hedged, bins=50, color='purple', edgecolor='k', alpha=0.5, density=True, label='Vega Hedge')
                        ax.axvline(-var_bs, color='blue', linestyle='--', label=f'VaR Original ({-var_bs:.2f})')
                        ax.axvline(-es_bs, color='blue', linestyle=':', label=f'ES Original ({-es_bs:.2f})')
                        ax.axvline(-var_vega_hedged, color='purple', linestyle='--', label=f'VaR Vega ({-var_vega_hedged:.2f})')
                        ax.axvline(-es_vega_hedged, color='purple', linestyle=':', label=f'ES Vega ({-es_vega_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Black-Scholes)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
                elif model == "Binomial":
                    value_binomial = bpa.portfolio_value(portfolio, N=N_steps_binomial)
                    sim_binomial = bpa.simulate_portfolio(portfolio, n_sims=10000, N=N_steps_binomial, horizon=horizon)
                    pnl_binomial = sim_binomial['pnl']
                    shocks_binomial = sim_binomial['shocks']
                    var_binomial, es_binomial = bpa.var_es(pnl_binomial, alpha=0.01)
                    if hedge_strategy == "Delta Hedge":
                        delta_hedge_fraction_binomial = coverage_percentage / 100.0
                        subyacentes = {}
                        for opt in portfolio:
                            key = opt.get('ticker', opt['S'])
                            greeks = bpa.option_greeks(opt, N=N_steps_binomial)
                            subyacentes.setdefault(key, {'S0': opt['S'], 'delta': 0})
                            subyacentes[key]['delta'] += greeks['delta'] * opt['qty'] * delta_hedge_fraction_binomial
                        pnl_binomial_hedged = []
                        for i in range(len(pnl_binomial)):
                            hedge_pnl = 0
                            for key, v in subyacentes.items():
                                S0 = v['S0']
                                delta = v['delta']
                                Z = shocks_binomial[key][i]
                                for opt in portfolio:
                                    if opt.get('ticker', opt['S']) == key:
                                        iv = bpa.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                        if iv is None:
                                            iv = 0.2
                                        r = opt['r']
                                        T_sim = horizon if horizon is not None else opt['T']
                                        break
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                hedge_pnl += -delta * (S_T - S0)
                            pnl_binomial_hedged.append(pnl_binomial[i] + hedge_pnl)
                        pnl_binomial_hedged = np.array(pnl_binomial_hedged)
                        var_binomial_hedged, es_binomial_hedged = bpa.var_es(pnl_binomial_hedged, alpha=0.01)
                        st.write(f"Delta hedge per underlying:")
                        for key, v in subyacentes.items():
                            st.write(f"  Underlying {key}: delta = {v['delta']:.4f}")
                        st.write(f"\nVaR after delta hedge (Binomial, 99%): {var_binomial_hedged:.2f}")
                        st.write(f"ES after delta hedge (Binomial, 99%): {es_binomial_hedged:.2f}")
                        st.write(f"VaR reduction: {var_binomial - var_binomial_hedged:.2f}")
                        st.write(f"ES reduction: {es_binomial - es_binomial_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_binomial, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_binomial_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
                        ax.axvline(-var_binomial, color='blue', linestyle='--', label=f'VaR Original ({-var_binomial:.2f})')
                        ax.axvline(-es_binomial, color='blue', linestyle=':', label=f'ES Original ({-es_binomial:.2f})')
                        ax.axvline(-var_binomial_hedged, color='red', linestyle='--', label=f'VaR Delta ({-var_binomial_hedged:.2f})')
                        ax.axvline(-es_binomial_hedged, color='red', linestyle=':', label=f'ES Delta ({-es_binomial_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Binomial)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
                    elif hedge_strategy == "Delta+Gamma Hedge":
                        greeks_total = bpa.portfolio_greeks(portfolio, N=N_steps_binomial)
                        gamma_cartera = greeks_total['gamma']
                        delta_cartera = greeks_total['delta']
                        hedge_opt = {
                            'type': 'call',
                            'style': 'european',
                            'S': portfolio[0]['S'],
                            'K': portfolio[0]['K'],
                            'T': portfolio[0]['T'],
                            'r': portfolio[0]['r'],
                            'qty': 0,
                            'market_price': portfolio[0]['market_price'],
                        }
                        greeks_hedge = bpa.option_greeks(hedge_opt, N=N_steps_binomial)
                        gamma_hedge = greeks_hedge['gamma']
                        gamma_hedge_fraction = coverage_percentage / 100.0
                        qty_gamma_hedge = -gamma_cartera * gamma_hedge_fraction / gamma_hedge if gamma_hedge != 0 else 0
                        hedge_opt['qty'] = qty_gamma_hedge
                        portfolio_gamma_hedged = portfolio + [hedge_opt]
                        greeks_total_gamma = bpa.portfolio_greeks(portfolio_gamma_hedged, N=N_steps_binomial)
                        delta_gamma_hedged = greeks_total_gamma['delta']
                        pnl_gamma_delta_hedged = []
                        for i in range(len(pnl_binomial)):
                            shocked_portfolio = []
                            for opt in portfolio_gamma_hedged:
                                key = opt.get('ticker', opt['S'])
                                S0 = opt['S']
                                iv = bpa.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                if iv is None:
                                    iv = 0.2
                                r = opt['r']
                                T_sim = horizon if horizon is not None else opt['T']
                                Z = shocks_binomial[key][i] if key in shocks_binomial else np.random.normal(0, 1)
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                shocked_opt = opt.copy()
                                shocked_opt['S'] = S_T
                                shocked_portfolio.append(shocked_opt)
                            val = bpa.portfolio_value(shocked_portfolio, N=N_steps_binomial)
                            hedge_pnl = 0
                            S0 = portfolio[0]['S']
                            delta = delta_gamma_hedged
                            Z = shocks_binomial[portfolio[0].get('ticker', portfolio[0]['S'])][i]
                            S_T = S0 * np.exp((portfolio[0]['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                            hedge_pnl += -delta * (S_T - S0)
                            pnl_gamma_delta_hedged.append(val - value_binomial + hedge_pnl)
                        pnl_gamma_delta_hedged = np.array(pnl_gamma_delta_hedged)
                        var_gamma_delta_hedged, es_gamma_delta_hedged = bpa.var_es(pnl_gamma_delta_hedged, alpha=0.01)
                        st.write(f"Gamma hedge: qty = {qty_gamma_hedge:.4f} of ATM call (S={hedge_opt['S']}, K={hedge_opt['K']}) covering {gamma_hedge_fraction*100:.0f}% of gamma")
                        st.write(f"Delta after gamma hedge: {delta_gamma_hedged:.4f}")
                        st.write(f"\nVaR after gamma+delta hedge (Binomial, 99%): {var_gamma_delta_hedged:.2f}")
                        st.write(f"ES after gamma+delta hedge (Binomial, 99%): {es_gamma_delta_hedged:.2f}")
                        st.write(f"VaR reduction vs original: {var_binomial - var_gamma_delta_hedged:.2f}")
                        st.write(f"ES reduction vs original: {es_binomial - es_gamma_delta_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_binomial, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_gamma_delta_hedged, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
                        ax.axvline(-var_binomial, color='blue', linestyle='--', label=f'VaR Original ({-var_binomial:.2f})')
                        ax.axvline(-es_binomial, color='blue', linestyle=':', label=f'ES Original ({-es_binomial:.2f})')
                        ax.axvline(-var_gamma_delta_hedged, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_gamma_delta_hedged:.2f})')
                        ax.axvline(-es_gamma_delta_hedged, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_gamma_delta_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Binomial)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
                    elif hedge_strategy == "Vega Hedge":
                        vega_total = bpa.portfolio_greeks(portfolio, N=N_steps_binomial)['vega']
                        hedge_opt_vega = {
                            'type': 'call',
                            'style': 'european',
                            'S': portfolio[0]['S'],
                            'K': portfolio[0]['K'],
                            'T': portfolio[0]['T'],
                            'r': portfolio[0]['r'],
                            'qty': 0,
                            'market_price': portfolio[0]['market_price'],
                        }
                        greeks_hedge_vega = bpa.option_greeks(hedge_opt_vega, N=N_steps_binomial)
                        vega_hedge = greeks_hedge_vega['vega']
                        vega_hedge_fraction = coverage_percentage / 100.0
                        qty_vega_hedge = -vega_total * vega_hedge_fraction / vega_hedge if vega_hedge != 0 else 0
                        hedge_opt_vega['qty'] = qty_vega_hedge
                        portfolio_vega_hedged = portfolio + [hedge_opt_vega]
                        pnl_vega_hedged = []
                        for i in range(len(pnl_binomial)):
                            shocked_portfolio = []
                            for opt in portfolio_vega_hedged:
                                key = opt.get('ticker', opt['S'])
                                S0 = opt['S']
                                iv = bpa.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                if iv is None:
                                    iv = 0.2
                                r = opt['r']
                                T_sim = horizon if horizon is not None else opt['T']
                                Z = shocks_binomial[key][i] if key in shocks_binomial else np.random.normal(0, 1)
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                shocked_opt = opt.copy()
                                shocked_opt['S'] = S_T
                                shocked_portfolio.append(shocked_opt)
                            val = bpa.portfolio_value(shocked_portfolio, N=N_steps_binomial)
                            pnl_vega_hedged.append(val - value_binomial)
                        pnl_vega_hedged = np.array(pnl_vega_hedged)
                        var_vega_hedged, es_vega_hedged = bpa.var_es(pnl_vega_hedged, alpha=0.01)
                        st.write(f"Vega hedge: qty = {qty_vega_hedge:.4f} of ATM call (S={hedge_opt_vega['S']}, K={hedge_opt_vega['K']}) covering {vega_hedge_fraction*100:.0f}% of vega")
                        st.write(f"\nVaR after vega hedge (Binomial, 99%): {var_vega_hedged:.2f}")
                        st.write(f"ES after vega hedge (Binomial, 99%): {es_vega_hedged:.2f}")
                        st.write(f"VaR reduction vs original: {var_binomial - var_vega_hedged:.2f}")
                        st.write(f"ES reduction vs original: {es_binomial - es_vega_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_binomial, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_vega_hedged, bins=50, color='purple', edgecolor='k', alpha=0.5, density=True, label='Vega Hedge')
                        ax.axvline(-var_binomial, color='blue', linestyle='--', label=f'VaR Original ({-var_binomial:.2f})')
                        ax.axvline(-es_binomial, color='blue', linestyle=':', label=f'ES Original ({-es_binomial:.2f})')
                        ax.axvline(-var_vega_hedged, color='purple', linestyle='--', label=f'VaR Vega ({-var_vega_hedged:.2f})')
                        ax.axvline(-es_vega_hedged, color='purple', linestyle=':', label=f'ES Vega ({-es_vega_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Binomial)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
                elif model == "Monte Carlo":
                    value_mc = sum(mca.price_option_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps) * opt['qty'] for opt in portfolio)
                    sim_mc = mca.simulate_portfolio_mc_pricing(portfolio, n_sims=n_sim_main, n_steps=N_steps, horizon=horizon)
                    pnl_mc = sim_mc['pnl']
                    shocks_mc = sim_mc['shocks']
                    var_mc, es_mc = mca.var_es(pnl_mc, alpha=0.01)
                    if hedge_strategy == "Delta Hedge":
                        delta_hedge_fraction_mc = coverage_percentage / 100.0
                        subyacentes = {}
                        for opt in portfolio:
                            key = opt.get('ticker', opt['S'])
                            greeks = mca.option_greeks_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps)
                            subyacentes.setdefault(key, {'S0': opt['S'], 'delta': 0})
                            subyacentes[key]['delta'] += greeks['delta'] * opt['qty'] * delta_hedge_fraction_mc
                        pnl_mc_hedged = []
                        for i in range(len(pnl_mc)):
                            hedge_pnl = 0
                            for key, v in subyacentes.items():
                                S0 = v['S0']
                                delta = v['delta']
                                Z = shocks_mc[key][i]
                                for opt in portfolio:
                                    if opt.get('ticker', opt['S']) == key:
                                        iv = mca.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                        if iv is None:
                                            iv = 0.2
                                        r = opt['r']
                                        T_sim = horizon if horizon is not None else opt['T']
                                        break
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                hedge_pnl += -delta * (S_T - S0)
                            pnl_mc_hedged.append(pnl_mc[i] + hedge_pnl)
                        pnl_mc_hedged = np.array(pnl_mc_hedged)
                        var_mc_hedged, es_mc_hedged = mca.var_es(pnl_mc_hedged, alpha=0.01)
                        st.write(f"Delta hedge per underlying:")
                        for key, v in subyacentes.items():
                            st.write(f"  Underlying {key}: delta = {v['delta']:.4f}")
                        st.write(f"\nVaR after delta hedge (MC, 99%): {var_mc_hedged:.2f}")
                        st.write(f"ES after delta hedge (MC, 99%): {es_mc_hedged:.2f}")
                        st.write(f"VaR reduction: {var_mc - var_mc_hedged:.2f}")
                        st.write(f"ES reduction: {es_mc - es_mc_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_mc, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_mc_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
                        ax.axvline(-var_mc, color='blue', linestyle='--', label=f'VaR Original ({-var_mc:.2f})')
                        ax.axvline(-es_mc, color='blue', linestyle=':', label=f'ES Original ({-es_mc:.2f})')
                        ax.axvline(-var_mc_hedged, color='red', linestyle='--', label=f'VaR Delta ({-var_mc_hedged:.2f})')
                        ax.axvline(-es_mc_hedged, color='red', linestyle=':', label=f'ES Delta ({-es_mc_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Monte Carlo)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
                    elif hedge_strategy == "Delta+Gamma Hedge":
                        greeks_total = mca.portfolio_greeks_mc(portfolio, n_sim=n_sim_greeks, n_steps=N_steps)
                        gamma_cartera = greeks_total['gamma']
                        delta_cartera = greeks_total['delta']
                        hedge_opt = {
                            'type': 'call',
                            'style': 'european',
                            'S': portfolio[0]['S'],
                            'K': portfolio[0]['K'],
                            'T': portfolio[0]['T'],
                            'r': portfolio[0]['r'],
                            'qty': 0,
                            'market_price': portfolio[0]['market_price'],
                        }
                        greeks_hedge = mca.option_greeks_mc(hedge_opt, n_sim=n_sim_greeks, n_steps=N_steps)
                        gamma_hedge = greeks_hedge['gamma']
                        gamma_hedge_fraction = coverage_percentage / 100.0
                        qty_gamma_hedge = -gamma_cartera * gamma_hedge_fraction / gamma_hedge if gamma_hedge != 0 else 0
                        hedge_opt['qty'] = qty_gamma_hedge
                        portfolio_gamma_hedged = portfolio + [hedge_opt]
                        greeks_total_gamma = mca.portfolio_greeks_mc(portfolio_gamma_hedged, n_sim=n_sim_greeks, n_steps=N_steps)
                        delta_gamma_hedged = greeks_total_gamma['delta']
                        pnl_gamma_delta_hedged = []
                        for i in range(len(pnl_mc)):
                            shocked_portfolio = []
                            for opt in portfolio_gamma_hedged:
                                key = opt.get('ticker', opt['S'])
                                S0 = opt['S']
                                iv = mca.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                if iv is None:
                                    iv = 0.2
                                r = opt['r']
                                T_sim = horizon if horizon is not None else opt['T']
                                Z = shocks_mc[key][i] if key in shocks_mc else np.random.normal(0, 1)
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                shocked_opt = opt.copy()
                                shocked_opt['S'] = S_T
                                shocked_portfolio.append(shocked_opt)
                            val = sum(mca.price_option_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps) * opt['qty'] for opt in shocked_portfolio)
                            hedge_pnl = 0
                            S0 = portfolio[0]['S']
                            delta = delta_gamma_hedged
                            Z = shocks_mc[portfolio[0].get('ticker', portfolio[0]['S'])][i]
                            S_T = S0 * np.exp((portfolio[0]['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                            hedge_pnl += -delta * (S_T - S0)
                            pnl_gamma_delta_hedged.append(val - value_mc + hedge_pnl)
                        pnl_gamma_delta_hedged = np.array(pnl_gamma_delta_hedged)
                        var_gamma_delta_hedged, es_gamma_delta_hedged = mca.var_es(pnl_gamma_delta_hedged, alpha=0.01)
                        st.write(f"Gamma hedge: qty = {qty_gamma_hedge:.4f} of ATM call (S={hedge_opt['S']}, K={hedge_opt['K']}) covering {gamma_hedge_fraction*100:.0f}% of gamma")
                        st.write(f"Delta after gamma hedge: {delta_gamma_hedged:.4f}")
                        st.write(f"\nVaR after gamma+delta hedge (MC, 99%): {var_gamma_delta_hedged:.2f}")
                        st.write(f"ES after gamma+delta hedge (MC, 99%): {es_gamma_delta_hedged:.2f}")
                        st.write(f"VaR reduction vs original: {var_mc - var_gamma_delta_hedged:.2f}")
                        st.write(f"ES reduction vs original: {es_mc - es_gamma_delta_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_mc, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_gamma_delta_hedged, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
                        ax.axvline(-var_mc, color='blue', linestyle='--', label=f'VaR Original ({-var_mc:.2f})')
                        ax.axvline(-es_mc, color='blue', linestyle=':', label=f'ES Original ({-es_mc:.2f})')
                        ax.axvline(-var_gamma_delta_hedged, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_gamma_delta_hedged:.2f})')
                        ax.axvline(-es_gamma_delta_hedged, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_gamma_delta_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Monte Carlo)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
                    elif hedge_strategy == "Vega Hedge":
                        vega_total = mca.portfolio_greeks_mc(portfolio, n_sim=n_sim_greeks, n_steps=N_steps)['vega']
                        hedge_opt_vega = {
                            'type': 'call',
                            'style': 'european',
                            'S': portfolio[0]['S'],
                            'K': portfolio[0]['K'],
                            'T': portfolio[0]['T'],
                            'r': portfolio[0]['r'],
                            'qty': 0,
                            'market_price': portfolio[0]['market_price'],
                        }
                        greeks_hedge_vega = mca.option_greeks_mc(hedge_opt_vega, n_sim=n_sim_greeks, n_steps=N_steps)
                        vega_hedge = greeks_hedge_vega['vega']
                        vega_hedge_fraction = coverage_percentage / 100.0
                        qty_vega_hedge = -vega_total * vega_hedge_fraction / vega_hedge if vega_hedge != 0 else 0
                        hedge_opt_vega['qty'] = qty_vega_hedge
                        portfolio_vega_hedged = portfolio + [hedge_opt_vega]
                        pnl_vega_hedged = []
                        for i in range(len(pnl_mc)):
                            shocked_portfolio = []
                            for opt in portfolio_vega_hedged:
                                key = opt.get('ticker', opt['S'])
                                S0 = opt['S']
                                iv = mca.implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                if iv is None:
                                    iv = 0.2
                                r = opt['r']
                                T_sim = horizon if horizon is not None else opt['T']
                                Z = shocks_mc[key][i] if key in shocks_mc else np.random.normal(0, 1)
                                S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                                shocked_opt = opt.copy()
                                shocked_opt['S'] = S_T
                                shocked_portfolio.append(shocked_opt)
                            val = sum(mca.price_option_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps) * opt['qty'] for opt in shocked_portfolio)
                            pnl_vega_hedged.append(val - value_mc)
                        pnl_vega_hedged = np.array(pnl_vega_hedged)
                        var_vega_hedged, es_vega_hedged = mca.var_es(pnl_vega_hedged, alpha=0.01)
                        st.write(f"Vega hedge: qty = {qty_vega_hedge:.4f} of ATM call (S={hedge_opt_vega['S']}, K={hedge_opt_vega['K']}) covering {vega_hedge_fraction*100:.0f}% of vega")
                        st.write(f"\nVaR after vega hedge (MC, 99%): {var_vega_hedged:.2f}")
                        st.write(f"ES after vega hedge (MC, 99%): {es_vega_hedged:.2f}")
                        st.write(f"VaR reduction vs original: {var_mc - var_vega_hedged:.2f}")
                        st.write(f"ES reduction vs original: {es_mc - es_vega_hedged:.2f}")
                        fig, ax = plt.subplots(figsize=(14, 8))
                        ax.hist(pnl_mc, bins=50, color='lightblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                        ax.hist(pnl_vega_hedged, bins=50, color='purple', edgecolor='k', alpha=0.5, density=True, label='Vega Hedge')
                        ax.axvline(-var_mc, color='blue', linestyle='--', label=f'VaR Original ({-var_mc:.2f})')
                        ax.axvline(-es_mc, color='blue', linestyle=':', label=f'ES Original ({-es_mc:.2f})')
                        ax.axvline(-var_vega_hedged, color='purple', linestyle='--', label=f'VaR Vega ({-var_vega_hedged:.2f})')
                        ax.axvline(-es_vega_hedged, color='purple', linestyle=':', label=f'ES Vega ({-es_vega_hedged:.2f})')
                        ax.set_title('P&L Distribution Comparison (Monte Carlo)')
                        ax.set_xlabel('P&L')
                        ax.set_ylabel('Density')
                        ax.legend()
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in calculation: {e}")
     # Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if menu1 == "Sensitivity Analysis":
    st.header("🔬 Sensitivity Analysis")
    st.subheader("Stress-test your portfolio against market shocks")
    st.write("Select your model and analyze how different factors affect your portfolio's performance")
    # Select model for hedging strategy
    model = st.selectbox("Select Model:", ["Black-Scholes", "Binomial", "Monte Carlo"], index=0)
    # Select percentage of coverage for hedging
    coverage_percentage = st.number_input("Hedging (%):", value=70, min_value=0, max_value=100, step=1, help="Percentage of the portfolio to be hedged.")
    # Select VaR horizon after selecting hedge
    horizon = st.number_input("Horizon (e.g., enter 10/252 for a 10-day horizon)", value=0.0397, min_value=0.01, format="%.4f", help="Horizon for VaR calculation.")
    num_options = st.number_input("Number of options in portfolio", min_value=1, max_value=10, value=4, step=1, help="Number of different options in the portfolio.")
    portfolio = []
    default_values = [
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5915, 'T': 0.0849, 'r': 0.0421, 'qty': -10, 'market_price': 111.93},
        {'type': 'put', 'style': 'american', 'S': 5912.17, 'K': 5910, 'T': 0.0849, 'r': 0.0421, 'qty': -5, 'market_price': 106.89},
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5920, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 103.66},
        {'type': 'put', 'style': 'european', 'S': 5912.17, 'K': 5900, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 102.92}
    ]
    
    # Add conditional logic for model-specific settings
    if model == "Monte Carlo":
        st.write("Monte Carlo model selected.")
        n_sim_main = st.number_input("Number of simulations for P&L and VaR/ES", value=1000, min_value=1000, step=1000, help="Number of scenarios for P&L and VaR/ES simulation.", key="n_sim_main_sensitivity")
        n_sim_greeks = st.number_input("Number of simulations for Greeks", value=1000, min_value=1000, step=1000, help="Number of scenarios for Greeks calculation.", key="n_sim_greeks_sensitivity")
        n_sim_sens = st.number_input("Number of simulations for Sensitivities", value=1000, min_value=1000, step=1000, help="Number of scenarios for sensitivity analysis.", key="n_sim_sens_sensitivity")
        st.write("Note: The following input uses the Longstaff-Schwartz method.")
        N_steps = st.number_input("Number of steps (For short maturities, use fewer steps; for long maturities, use more steps)", value=100, min_value=1, step=1, help="Discretization steps for Monte Carlo model.")
    elif model == "Black-Scholes":
        st.write("Black-Scholes model selected.")
        # Add Black-Scholes specific inputs here
    elif model == "Binomial":
        st.write("Binomial model selected.")
        N_steps_binomial = st.number_input("Number of steps for Binomial tree", value=100, min_value=1, step=1, help="Set the discretization steps for the Binomial model.", key="N_steps_binomial_sensitivity")
    
    # Ensure portfolio options are customizable
    for i in range(num_options):
        st.subheader(f"Option {i+1}")
        option_type = st.selectbox(f"Option type for Option {i+1}", ["call", "put"], index=["call", "put"].index(default_values[i]['type']), key=f"option_type_{i}", help="Call or put option.")
        option_style = st.selectbox(f"Option style for Option {i+1}", ["european", "american"], index=["european", "american"].index(default_values[i]['style']), key=f"option_style_{i}", help="Exercise style of the option.")
        S = st.number_input(f"Spot price (S) for Option {i+1}", value=default_values[i]['S'], help="Current price of the underlying asset.", key=f"S_{i}")
        K = st.number_input(f"Strike price (K) for Option {i+1}", value=default_values[i]['K'], help="Strike price of the option.", key=f"K_{i}")
        T = st.number_input(f"Time to maturity (years) for Option {i+1}", value=default_values[i]['T'], min_value=0.01, format="%.4f", help="Time to maturity in years.", key=f"T_{i}")
        r = st.number_input(f"Risk-free rate (r, decimal) for Option {i+1}", value=default_values[i]['r'], min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", help="Annual risk-free interest rate.", key=f"r_{i}")
        qty = st.number_input(f"Quantity for Option {i+1}", value=default_values[i]['qty'], step=1, help="Quantity of options in the portfolio.", key=f"qty_{i}")
        market_price = st.number_input(f"Market price for Option {i+1}", value=default_values[i]['market_price'], help="Observed market price of the option.", key=f"market_price_{i}")
        portfolio.append({'type': option_type, 'style': option_style, 'S': S, 'K': K, 'T': T, 'r': r, 'qty': qty, 'market_price': market_price})

    if st.button("Calculate Sensitivity Analysis", key="hedging_strategy_btn"):
        with st.spinner("Calculating hedging strategy and sensitivity analysis..."):
            try:
                if model == "Black-Scholes":
                    # Run sensitivity analysis for Black-Scholes
                    bsa.run_sensitivity_analysis_bs(portfolio, bsa.VIS_DIR, selected_strategy='all')  # Provide default value for selected_strategy
                    st.success("Sensitivity analysis completed. Check the visualizations directory for results.")
                    # Display graphs and numerical tables
                    st.image(os.path.join(bsa.VIS_DIR, 'sensitivity_spot_all_bs.png'))
                    st.image(os.path.join(bsa.VIS_DIR, 'sensitivity_r_all_bs.png'))
                    st.image(os.path.join(bsa.VIS_DIR, 'sensitivity_vol_all_bs.png'))
                    st.write("Numerical tables:")
                    st.write(pd.read_csv(os.path.join(bsa.VIS_DIR, 'sensitivity_spot_all_bs.csv')))
                    st.write(pd.read_csv(os.path.join(bsa.VIS_DIR, 'sensitivity_r_all_bs.csv')))
                    st.write(pd.read_csv(os.path.join(bsa.VIS_DIR, 'sensitivity_vol_all_bs.csv')))
                elif model == "Monte Carlo":
                    vis_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    mca.run_sensitivity_analysis_mc(portfolio, N_steps, n_sim_greeks, bsa.VIS_DIR, horizon)
                    st.success("Sensitivity analysis completed. Check the visualizations directory for results.")
                    # Display graphs and numerical tables
                    st.image(os.path.join(bsa.VIS_DIR, 'sensitivity_spot_mc.png'))
                    st.image(os.path.join(bsa.VIS_DIR, 'sensitivity_r_mc.png'))
                    st.image(os.path.join(bsa.VIS_DIR, 'sensitivity_vol_mc.png'))
                    st.write("Numerical tables:")
                    st.write(pd.read_csv(os.path.join(bsa.VIS_DIR, 'sensitivity_spot_mc.csv')))
                    st.write(pd.read_csv(os.path.join(bsa.VIS_DIR, 'sensitivity_r_mc.csv')))
                    st.write(pd.read_csv(os.path.join(bsa.VIS_DIR, 'sensitivity_vol_mc.csv')))
                elif model == "Binomial":
                    bpa.run_sensitivity_analysis_binomial(portfolio, N_steps_binomial)
                    st.success("Sensitivity analysis completed. Check the visualizations directory for results.")
                    # Display graphs and numerical tables
                    st.image(os.path.join(bpa.VIS_DIR, 'sensitivity_spot_all.png'))
                    st.image(os.path.join(bpa.VIS_DIR, 'sensitivity_r_all.png'))
                    st.image(os.path.join(bpa.VIS_DIR, 'sensitivity_vol_all.png'))
                    st.write("Numerical tables:")
                    st.write(pd.read_csv(os.path.join(bpa.VIS_DIR, 'sensitivity_spot_all.csv')))
                    st.write(pd.read_csv(os.path.join(bpa.VIS_DIR, 'sensitivity_r_all.csv')))
                    st.write(pd.read_csv(os.path.join(bpa.VIS_DIR, 'sensitivity_vol_all.csv')))
            except Exception as e:
                st.error(f"Error in calculation: {e}")
            except Exception as e:
                st.error(f"Error in sensitivity analysis calculation: {e}")
     # Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
if menu1 == "Technical Analysis":
    st.header("📈 Technical Analysis")
    st.write("Perform technical analysis on stock data.")
    # User input for ticker and interval
    ticker = st.text_input("Enter Stock Ticker:", "AAPL")
    interval = st.selectbox("Select Interval:", ["daily", "weekly", "monthly"])
    start_date = st.date_input("Select Start Date:", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("Select End Date:", pd.to_datetime("2023-01-01"))

    # Download data
    df = descargar_datos(ticker=ticker, interval=interval, start=start_date, end=end_date)

    # Single button to perform all calculations and plots
    if st.button("Run Full Technical Analysis"):
        # Perform all calculations
        calcular_sma_multiple(df)
        calcular_ema_multiple(df)
        calcular_rsi(df)
        calcular_macd(df)
        calcular_bollinger_bands(df)
        calcular_momentum(df)
        calcular_adx(df)
        calcular_obv(df)
        calcular_stochastic_oscillator(df)

        # Sort DataFrame by date in descending order
        df_sorted = df.sort_values(by='date', ascending=False)

        # Display the last 5 most recent dates
        st.write("SMA calculated:", df_sorted[['date', 'close', 'sma_20', 'sma_50', 'sma_200']].head(5))
        st.write("EMA calculated:", df_sorted[['date', 'close', 'ema_20', 'ema_50']].head(5))
        st.write("RSI calculated:", df_sorted[['date', 'close', 'rsi']].head(5))
        st.write("MACD calculated:", df_sorted[['date', 'close', 'macd', 'signal_line']].head(5))
        st.write("Bollinger Bands calculated:", df_sorted[['date', 'close', 'upper_band', 'lower_band']].head(5))
        st.write("Momentum calculated:", df_sorted[['date', 'close', 'momentum']].head(5))
        st.write("ADX calculated:", df_sorted[['date', 'close', 'adx']].head(5))
        st.write("OBV calculated:", df_sorted[['date', 'close', 'obv']].head(5))
        st.write("Stochastic Oscillator calculated:", df_sorted[['date', 'close', 'stoch_k', 'stoch_d']].head(5))

        # Generate and display plots
        fig1 = plot_candlestick_and_momentum(df, ticker)
        st.pyplot(fig1)

        fig2 = plot_candlestick_and_rsi(df, ticker)
        st.pyplot(fig2)

        fig3 = plot_candlestick_and_macd(df, ticker)
        st.pyplot(fig3)

        fig4 = plot_candlestick_and_bollinger(df, ticker)
        st.pyplot(fig4)

        fig5 = plot_sma_multiple(df, ticker)
        st.pyplot(fig5)

        fig6 = plot_ema_multiple(df, ticker)
        st.pyplot(fig6)

        fig7 = plot_adx(df, ticker)
        st.pyplot(fig7)

        fig8 = plot_stochastic_oscillator(df, ticker)
        st.pyplot(fig8)

        fig9 = plot_macd_with_adx(df, ticker)
        st.pyplot(fig9)

        fig10 = plot_macd_with_stochastic(df, ticker)
        st.pyplot(fig10)

        fig11 = plot_rsi_with_adx(df, ticker)
        st.pyplot(fig11)

        fig12 = plot_rsi_with_stochastic(df, ticker)
        st.pyplot(fig12)

    # Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if menu1 == "Fundamental Analysis":
    st.header("📊 Fundamental Analysis")
    st.subheader("Analyze the financial health of a company")
    
    # User input for ticker symbol
    ticker = st.text_input("Enter Ticker Symbol:", "META")
    
    # Button to trigger analysis
    if st.button("Run Analysis"):
        # Remove old visualizations
        files = glob.glob(os.path.join('portfolio_management/visualizations', '*.png'))
        for f in files:
            os.remove(f)
        
        # Initialize FinancialAnalyzer
        analyzer = FinancialAnalyzer(ticker)
        
        # Core financial statements
        st.subheader("Balance Sheet")
        figures = analyzer.get_balance_sheet(plot=True)
        for fig in figures:
            st.pyplot(fig)
        
        # Display Balance Sheet CSV with formatting
        balance_sheet_df = pd.read_csv(os.path.join(analyzer.VIS_DIR, 'balance_sheet.csv'))
        # Remove any column starting with '2020'
        balance_sheet_df = balance_sheet_df.loc[:, ~balance_sheet_df.columns.str.startswith('2020')]
        balance_sheet_df_formatted = balance_sheet_df.applymap(lambda x: f"{x:,.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else x)
        st.dataframe(balance_sheet_df_formatted)
        
        st.subheader("Income Statement")
        figures = analyzer.get_income_statement(plot=True)
        for fig in figures:
            st.pyplot(fig)
        
        # Display Income Statement CSV with formatting
        income_statement_df = pd.read_csv(os.path.join(analyzer.VIS_DIR, 'income_statement.csv'))
        # Remove any column starting with '2020'
        income_statement_df = income_statement_df.loc[:, ~income_statement_df.columns.str.startswith('2020')]
        income_statement_df_formatted = income_statement_df.applymap(lambda x: f"{x:,.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else x)
        st.dataframe(income_statement_df_formatted)
        
        st.subheader("Cash Flow Statement")
        figures = analyzer.get_cash_flow(plot=True)
        for fig in figures:
            st.pyplot(fig)
        
        # Display Cash Flow Statement CSV with formatting
        cash_flow_df = pd.read_csv(os.path.join(analyzer.VIS_DIR, 'cash_flow.csv'))
        # Remove any column starting with '2020'
        cash_flow_df = cash_flow_df.loc[:, ~cash_flow_df.columns.str.startswith('2020')]
        cash_flow_df_formatted = cash_flow_df.applymap(lambda x: f"{x:,.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else x)
        st.dataframe(cash_flow_df_formatted)
        
        # Additional metrics and analysis
        st.subheader("Financial Ratios")
        financial_ratios = analyzer.get_financial_ratios()
        for category, ratio_dict in financial_ratios.items():
            st.write(f"**{category}:**")
            for ratio, value in ratio_dict.items():
                if value is not None:
                    if isinstance(value, str) and '%' in value:
                        st.write(f"- {ratio}: {value}")
                    else:
                        st.write(f"- {ratio}: {value}")
                else:
                    st.write(f"- {ratio}: N/A")
        
        additional_metrics = analyzer.get_additional_metrics()
        st.subheader("Additional Metrics")
        for metric, value in additional_metrics.items():
            st.write(f"- {metric}: {value if value is not None else 'N/A'}")
        
        # Growth Metrics
        st.subheader("Growth Metrics")
        growth_metrics = analyzer.get_growth_metrics()
        for metric, value in growth_metrics.items():
            st.write(f"- {metric}: {value}")

        # Trend Analysis
        st.subheader("Trend Analysis")
        trend_analysis = analyzer.get_trend_analysis()
        for metric, value in trend_analysis.items():
            st.write(f"- {metric}: {value}")
        
        dividend_analysis = analyzer.get_dividend_analysis()
        st.subheader("Dividend Analysis")
        for metric, value in dividend_analysis.items():
            if metric == 'Dividend History' and isinstance(value, dict):
                st.write(f"**{metric}:**")
                for date, amount in value.items():
                    st.write(f"- {date}: {amount}")
            else:
                st.write(f"- {metric}: {value}")
        
        risk_metrics = analyzer.get_risk_metrics()
        st.subheader("Risk Metrics")
        for metric, value in risk_metrics.items():
            st.write(f"- {metric}: {value if value is not None else 'N/A'}")

# Personal info card below the sidebar menu
    st.sidebar.markdown('''
    <div style="background-color:#23272b; border-radius:12px; padding:1.2em 1.2em 1em 1.2em; margin-top:1.5em; margin-bottom:1.5em; box-shadow:0 2px 8px rgba(0,0,0,0.15); max-width:320px;">
        <div style="font-size:1.1rem; font-weight:700; color:#90caf9; margin-bottom:0.2em;">Marcos Heredia Pimienta</div>
        <div style="color:#b0bec5; font-size:0.98rem; margin-bottom:0.4em;">MSc in Quantitative Finance, Universitat Autònoma de Barcelona</div>
        <div style="color:#e0e0e0; font-size:0.95rem; margin-bottom:0.4em;">Quantitative Risk Analyst</div>
    </div>
    ''', unsafe_allow_html=True)

    # Enlace al formulario de Google
    st.sidebar.markdown('<div style="background-color:#23272b; padding:20px; border-radius:10px; margin-top:20px;">', unsafe_allow_html=True)
    st.sidebar.header("💬 We value your feedback!", anchor=None)
    st.sidebar.write("Please let us know how you feel about the app. Your insights help us improve!")

    # Hipervínculo al formulario
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSecDfBXdXynYHyouLub1ZT3AsYWa4V1N3O_OnvUKxiA21bnjg/viewform?usp=header"
    st.sidebar.markdown(f"[Fill out the survey]({form_url})", unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="footer-conference">Developed by Marcos Heredia Pimienta, Quantitative Risk Analyst</div>', unsafe_allow_html=True)


