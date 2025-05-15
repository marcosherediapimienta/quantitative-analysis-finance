import streamlit as st
import os
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt
import io
import yfinance as yf
from datetime import datetime
from scipy.optimize import brentq
import scipy.stats as stats

# ========== IMPORTACIÓN DE MODELOS ==========
sys.path.append(os.path.dirname(__file__))
from binomial_model.european_options.binomial import binomial_european_option_price, binomial_greeks_european_option
from finite_difference_method.european_options.call import finite_difference_european_call, finite_difference_greeks_call
from finite_difference_method.european_options.put import finite_difference_european_put, finite_difference_greeks_put

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Monte Carlo
mc_call_path = os.path.join(APP_DIR, "monte-carlo", "european_options", "call.py")
mc_put_path = os.path.join(APP_DIR, "monte-carlo", "european_options", "put.py")
mc_call_mod = import_module_from_path("mc_call_mod", mc_call_path)
mc_put_mod = import_module_from_path("mc_put_mod", mc_put_path)
monte_carlo_european_call = mc_call_mod.monte_carlo_european_call
monte_carlo_greeks_call = mc_call_mod.monte_carlo_greeks_call
monte_carlo_european_put = mc_put_mod.monte_carlo_european_put
monte_carlo_greeks_put = mc_put_mod.monte_carlo_greeks_put

# Black-Scholes
CALL_IV_PATH = os.path.join(APP_DIR, "black_scholes_model", "european_options", "call_implied_volatility.py")
PUT_IV_PATH = os.path.join(APP_DIR, "black_scholes_model", "european_options", "put_implied_volatility.py")
call_iv = import_module_from_path("call_iv", CALL_IV_PATH)
put_iv = import_module_from_path("put_iv", PUT_IV_PATH)

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def black_scholes_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

# ========== UI PRINCIPAL ==========
st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")
st.title("Option Pricing Dashboard")

with st.sidebar:
    st.header("Parámetros de la opción")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL")
    expiry = st.text_input("Vencimiento (YYYY-MM-DD)", value="2025-06-20")
    option_type = st.selectbox("Tipo de opción", ["call", "put"])
    K = st.number_input("Strike (K)", value=180.0)
    r = st.number_input("Tasa libre de riesgo (r, decimal)", value=0.0421)
    sigma = st.number_input("Volatilidad (σ, 0 para usar IV)", value=0.0)
    N = st.number_input("Steps Binomial/FD", value=100, min_value=10, step=10)
    M = st.number_input("Steps precio (FD)", value=100, min_value=10, step=10)
    calcular = st.button("Calcular y comparar modelos")

# ========== LÓGICA Y RESULTADOS ==========
if calcular:
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1d")
        if hist.empty:
            st.error("No se pudo obtener el precio spot para el ticker.")
            st.stop()
        S = hist['Close'].iloc[-1]
        try:
            options = data.option_chain(expiry)
        except Exception:
            st.error("No se pudo obtener la cadena de opciones para ese vencimiento.")
            st.stop()
        if option_type == 'call':
            row = options.calls[options.calls['strike'] == K]
        else:
            row = options.puts[options.puts['strike'] == K]
        if not row.empty:
            market_price = float(row['lastPrice'].iloc[0])
        else:
            market_price = None
        today = datetime.now().date()
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = (expiry_date - today).days / 365.0
        st.write(f"**Spot price (S):** {S:.2f}")
        st.write(f"**Strike (K):** {K}")
        st.write(f"**Expiry:** {expiry}")
        st.write(f"**Time to expiry (T):** {T:.4f} years")
        st.write(f"**Risk-free rate (r):** {r}")
        if market_price:
            st.write(f"**Market price:** {market_price}")

        def implied_volatility(option_type, market_price, S, K, T, r):
            def bs_price(sigma):
                if option_type == 'call':
                    return black_scholes_call_price(S, K, T, r, sigma)
                else:
                    return black_scholes_put_price(S, K, T, r, sigma)
            def objective(sigma):
                return bs_price(sigma) - market_price
            try:
                return brentq(objective, 1e-6, 5.0)
            except Exception:
                return np.nan

        if sigma == 0.0 and market_price:
            iv = implied_volatility(option_type, market_price, S, K, T, r)
            st.write(f"**Implied Volatility (IV):** {iv*100:.2f}%")
            sigma = iv
        else:
            iv = sigma
        # Black-Scholes
        if option_type == 'call':
            bs_price = black_scholes_call_price(S, K, T, r, sigma)
            greeks_bs = call_iv.calculate_greeks(S, K, T, r, sigma)
        else:
            bs_price = black_scholes_put_price(S, K, T, r, sigma)
            greeks_bs = put_iv.calculate_greeks(S, K, T, r, sigma)
        # Binomial
        binom_price = binomial_european_option_price(S, K, T, r, sigma, int(N), option_type)
        greeks_binom = binomial_greeks_european_option(S, K, T, r, sigma, int(N), option_type)
        # Finite Difference
        if option_type == 'call':
            fd_price = finite_difference_european_call(S, K, T, r, sigma, Smax=3, M=int(M), N=int(N))
            greeks_fd = finite_difference_greeks_call(S, K, T, r, sigma, Smax=3, M=int(M), N=int(N))
        else:
            fd_price = finite_difference_european_put(S, K, T, r, sigma, Smax=3, M=int(M), N=int(N))
            greeks_fd = finite_difference_greeks_put(S, K, T, r, sigma, Smax=3, M=int(M), N=int(N))
        # Monte Carlo
        if option_type == 'call':
            mc_price = monte_carlo_european_call(S, K, T, r, sigma)
            greeks_mc = monte_carlo_greeks_call(S, K, T, r, sigma)
        else:
            mc_price = monte_carlo_european_put(S, K, T, r, sigma)
            greeks_mc = monte_carlo_greeks_put(S, K, T, r, sigma)
        # ========== TABS DE RESULTADOS ==========
        tabs = st.tabs(["Black-Scholes", "Binomial", "Diferencias Finitas", "Monte Carlo", "Comparación"])
        with tabs[0]:
            st.subheader("Black-Scholes")
            st.write(f"Precio: {bs_price:.4f}")
            st.write("Greeks:")
            st.json(greeks_bs)
        with tabs[1]:
            st.subheader("Binomial")
            st.write(f"Precio: {binom_price:.4f}")
            st.write("Greeks:")
            st.json(greeks_binom)
        with tabs[2]:
            st.subheader("Diferencias Finitas")
            st.write(f"Precio: {fd_price:.4f}")
            st.write("Greeks:")
            st.json(greeks_fd)
        with tabs[3]:
            st.subheader("Monte Carlo")
            st.write(f"Precio: {mc_price:.4f}")
            st.write("Greeks:")
            st.json(greeks_mc)
        with tabs[4]:
            st.subheader("Comparación de precios")
            prices = {
                'Black-Scholes': bs_price,
                'Binomial': binom_price,
                'Diferencias Finitas': fd_price,
                'Monte Carlo': mc_price
            }
            fig, ax = plt.subplots()
            ax.bar(prices.keys(), prices.values(), color=['gray', 'orange', 'blue', 'green'])
            ax.set_ylabel('Precio de la opción')
            ax.set_title('Comparación de precios teóricos')
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al obtener datos o calcular modelos: {e}")

st.markdown("""
---
Desarrollado por **Marcos Heredia Pimienta, Quantitative Risk Analyst**. Última actualización: Mayo 2025
""")
