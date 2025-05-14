import streamlit as st
import os
import time
from datetime import datetime
import importlib.util
import sys
import math

st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")
st.title("Option Pricing Dashboard")

st.markdown("""
This application calculates implied volatility for European options using the Black-Scholes model through numerical methods including Newton-Raphson and Brent's 
algorithm. It also computes theoretical option prices via Monte Carlo simulation, modeling stochastic processes while specifically supporting 
European-style exercise. The tool provides complete Greeks analysis (Delta, Gamma, Vega, Theta, Rho) with customizable parameters for strikes, 
expirations, and risk-free rates.


*Developed by Marcos Heredia Pimienta*
""")

# Project info section
with st.sidebar.expander("ℹ️ About the project", expanded=False):
    st.markdown("""
    Quantitative analysis project for financial options.
    Includes implied volatility calculation using the Black-Scholes model.
    """)
    st.markdown("Repository: [GitHub](https://github.com/marcosherediapimienta/quantitative-analysis-finance)")
    st.markdown("""
    ---
    **Author:** Marcos Heredia Pimienta  
    **Role:** Quantitative Risk Analyst
    """)

# Personal brand section in the sidebar
with st.sidebar.expander("👤 About Me", expanded=True):
    st.markdown("""
    ### Marcos Heredia Pimienta
    **Quantitative Risk Analyst**

    - Specialist in derivatives, risk management, financial modeling and financial forcasting

    _Let's connect and create value through quantitative thinking!_
    """)

# Utility to import Black-Scholes functions from existing scripts
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CALL_IV_PATH = os.path.join(APP_DIR, "black_scholes_model", "european_options", "call", "call_implied_volatility.py")
spec_call = importlib.util.spec_from_file_location(
    "call_iv", CALL_IV_PATH
)
call_iv = importlib.util.module_from_spec(spec_call)
sys.modules["call_iv"] = call_iv
spec_call.loader.exec_module(call_iv)

PUT_IV_PATH = os.path.join(APP_DIR, "black_scholes_model", "european_options", "put", "put_implied_volatility.py")
spec_put = importlib.util.spec_from_file_location(
    "put_iv", PUT_IV_PATH
)
put_iv = importlib.util.module_from_spec(spec_put)
sys.modules["put_iv"] = put_iv
spec_put.loader.exec_module(put_iv)

# Black-Scholes and Monte Carlo menu for implied volatility and pricing
st.header("Option Pricing: Black-Scholes & Monte Carlo")
tabs = st.tabs(["Black-Scholes (IV)", "Monte Carlo"])

with tabs[0]:
    option_type = st.selectbox("Option type", ["call", "put"], key="iv_type")
    S = st.number_input("Spot price (S)", value=100.0, min_value=0.01, key="iv_S")
    K = st.number_input("Strike (K)", value=100.0, min_value=0.01, key="iv_K")
    T = st.number_input("Time to maturity (years, T)", value=0.5, min_value=0.01, step=0.01, key="iv_T")
    r = st.number_input("Risk-free rate (r)", value=0.03, min_value=0.0, step=0.001, format="%.4f", key="iv_r")
    market_price = st.number_input("Option market price", value=10.0, min_value=0.0001, step=0.01, format="%.4f", key="iv_market")
    if st.button("Calculate Implied Volatility", key="iv_btn"):
        if option_type == "call":
            iv = call_iv.implied_volatility_newton(market_price, S, K, T, r)
            if iv is not None:
                st.success(f"Implied volatility: {iv*100:.2f}%")
                greeks = call_iv.calculate_greeks(S, K, T, r, iv)
                st.write("**Greeks:**")
                st.json(greeks)
                call_iv.plot_greeks(S, K, T, r, iv)
                img_path = os.path.join(APP_DIR, "call_greeks_analysis.png")
                if os.path.exists(img_path):
                    st.image(img_path, caption="Greeks plots (Call)", use_container_width=True)
                else:
                    st.info("No Greeks plot found for Call option.")
            else:
                st.error("Could not calculate implied volatility (did not converge)")
        else:
            iv = put_iv.implied_volatility_newton(market_price, S, K, T, r)
            if iv is not None:
                st.success(f"Implied volatility: {iv*100:.2f}%")
                greeks = put_iv.calculate_greeks(S, K, T, r, iv)
                st.write("**Greeks:**")
                st.json(greeks)
                put_iv.plot_greeks(S, K, T, r, iv)
                img_path = os.path.join(APP_DIR, "put_greeks_analysis.png")
                if os.path.exists(img_path):
                    st.image(img_path, caption="Greeks plots (Put)", use_container_width=True)
                else:
                    st.info("No Greeks plot found for Put option.")
            else:
                st.error("Could not calculate implied volatility (did not converge)")

with tabs[1]:
    st.subheader("Monte Carlo European Option Pricing")
    option_type_mc = st.selectbox("Option type", ["call", "put"], key="mc_type")
    S_mc = st.number_input("Spot price (S)", value=100.0, min_value=0.01, key="mc_S")
    K_mc = st.number_input("Strike (K)", value=100.0, min_value=0.01, key="mc_K")
    T_mc = st.number_input("Time to maturity (years, T)", value=0.5, min_value=0.01, step=0.01, key="mc_T")
    r_mc = st.number_input("Risk-free rate (r)", value=0.03, min_value=0.0, step=0.001, format="%.4f", key="mc_r")
    sigma_mc = st.number_input("Volatility (σ)", value=0.2, min_value=0.0001, step=0.01, format="%.4f", key="mc_sigma")
    N_mc = st.number_input("Number of Monte Carlo simulations", value=50000, min_value=1000, step=1000, key="mc_N")
    if st.button("Calculate Monte Carlo Price", key="mc_btn"):
        import numpy as np
        Z = np.random.standard_normal(int(N_mc))
        ST = S_mc * np.exp((r_mc - 0.5 * sigma_mc**2) * T_mc + sigma_mc * np.sqrt(T_mc) * Z)
        if option_type_mc == 'call':
            payoff = np.maximum(ST - K_mc, 0)
        else:
            payoff = np.maximum(K_mc - ST, 0)
        option_price = np.exp(-r_mc * T_mc) * np.mean(payoff)
        st.success(f"Monte Carlo estimated price: {option_price:.4f}")
        # Histogram of final prices
        import matplotlib.pyplot as plt
        import io
        fig, ax = plt.subplots()
        ax.hist(ST, bins=50, alpha=0.7)
        ax.set_title("Distribution of Final Prices at Maturity")
        ax.set_xlabel("Final Price")
        ax.set_ylabel("Frequency")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        st.image(buf, caption="Histogram of Simulated Final Prices", use_container_width=True)

st.markdown("""
---
Developed by **Marcos Heredia Pimienta, Quantitative Risk Analyst**. Last update: May 2025
""")
