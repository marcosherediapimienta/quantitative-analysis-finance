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
This application allows you to calculate the implied volatility of financial options using the Black-Scholes model.

*Developed by Marcos Heredia Pimienta*
""")

# Enhanced help section
with st.sidebar.expander("ℹ️ Help & Instructions", expanded=False):
    st.markdown("""
    **How to use the app?**
    - Enter the option parameters and the market price.
    - Click the button to calculate the implied volatility.
    - If an error occurs, the corresponding message will be displayed.
    - You can reload the page to clear the output.
    
    **Requirements:**
    - Python 3.8+
    - Install dependencies with: `pip install streamlit numpy matplotlib pandas scipy`
    - Run the app with: `streamlit run app.py`
    """)
    st.markdown("""
    **Common issues?**
    - If you see dependency errors, make sure they are installed.
    """)
    st.markdown("""
    **Contact:**
    - For support, contact your developer or check the project README.
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
    **Role:** Quantitative Analyst
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

# Black-Scholes menu for implied volatility only
st.header("Implied Volatility (Black-Scholes)")
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
            # Calculate and show greeks and plot
            greeks = call_iv.calculate_greeks(S, K, T, r, iv)
            st.write("**Greeks:**")
            st.json(greeks)
            call_iv.plot_greeks(S, K, T, r, iv)
            img_path = os.path.join(APP_DIR, "call_greeks_analysis.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Greeks plots (Call)", use_column_width=True)
        else:
            st.error("Could not calculate implied volatility (did not converge)")
    else:
        iv = put_iv.implied_volatility_newton(market_price, S, K, T, r)
        if iv is not None:
            st.success(f"Implied volatility: {iv*100:.2f}%")
            # Calculate and show greeks and plot
            greeks = put_iv.calculate_greeks(S, K, T, r, iv)
            st.write("**Greeks:**")
            st.json(greeks)
            put_iv.plot_greeks(S, K, T, r, iv)
            img_path = os.path.join(APP_DIR, "put_greeks_analysis.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Greeks plots (Put)", use_column_width=True)
        else:
            st.error("Could not calculate implied volatility (did not converge)")

st.markdown("""
---
Developed by **Marcos Heredia Pimienta, Quantitative Risk Analyst**. Last update: May 2025
""")
