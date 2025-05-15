import streamlit as st

st.set_page_config(page_title="European Option Pricing App", layout="centered")

st.markdown("""
<style>
    .main, .stApp, .css-18e3th9 {
        background-color: #181a1b;
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
</style>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "Select section:",
    [
        "Introduction",
        "Black-Scholes",
        "Binomial",
        "Monte Carlo",
        "Finite Differences",
        "Model Comparison"
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

if menu == "Introduction":
    st.markdown('<div class="title-conference">European Option Pricing App</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-conference">A robust and visual tool for calculating implied volatility and theoretical prices of European options using quantitative models.</div>', unsafe_allow_html=True)
    st.markdown("""
    **Instructions:**
    - Select a pricing model from the sidebar.
    - Enter the required parameters.
    - View results and compare models visually.
    """)
    st.markdown("""
    **About:**
    This application is designed for quantitative finance professionals, students, and researchers. It provides a unified interface to explore and compare the most important models for European option pricing, including their Greeks and implied volatility.
    """)

# Attempt to import the implied volatility modules
bs_import_error = None
try:
    from black_scholes_model.european_options.call_implied_volatility import implied_volatility_call
    from black_scholes_model.european_options.put_implied_volatility import implied_volatility_put
except Exception as e:
    bs_import_error = str(e)

if menu == "Black-Scholes":
    if bs_import_error:
        st.error(f"Error importing Black-Scholes functions: {bs_import_error}")
    else:
        st.write("Calculate the implied volatility and Greeks of a European option using the Black-Scholes model.")
        option_type = st.selectbox("Option type", ["call", "put"])
        S = st.number_input("Spot price (S)", value=100.0000, format="%.4f")
        K = st.number_input("Strike price (K)", value=100.0000, format="%.4f")
        T = st.number_input("Time to maturity (years)", value=1.0000, format="%.4f")
        r = st.number_input("Risk-free rate (r, decimal)", value=0.0500, format="%.4f")
        market_price = st.number_input("Option market price", value=10.0000, format="%.4f")
        if st.button("Calculate Implied Volatility", key="bs_btn"):
            try:
                if option_type == "call":
                    iv = implied_volatility_call(S, K, T, r, market_price)
                    from black_scholes_model.european_options.call_implied_volatility import calculate_greeks
                    greeks = calculate_greeks(S, K, T, r, iv)
                else:
                    iv = implied_volatility_put(S, K, T, r, market_price)
                    from black_scholes_model.european_options.put_implied_volatility import calculate_greeks
                    greeks = calculate_greeks(S, K, T, r, iv)
                st.success(f"Implied Volatility ({option_type}): {iv:.4f}")
                st.write("Greeks:")
                st.json(greeks)
            except Exception as e:
                st.error(f"Error calculating implied volatility: {e}")

elif menu == "Binomial":
    try:
        from binomial_model.european_options.binomial import implied_volatility_option, binomial_european_option_price, binomial_greeks_european_option
    except Exception as e:
        st.error(f"Error importing Binomial model: {e}")
    else:
        st.write("Calculate the theoretical price and Greeks of a European option using the Binomial method.")
        option_type = st.selectbox("Option type", ["call", "put"], key="bin_type")
        S = st.number_input("Spot price (S)", value=100.0000, key="bin_s", format="%.4f")
        K = st.number_input("Strike price (K)", value=100.0000, key="bin_k", format="%.4f")
        T = st.number_input("Time to maturity (years)", value=1.0000, key="bin_t", format="%.4f")
        r = st.number_input("Risk-free rate (r, decimal)", value=0.0500, key="bin_r", format="%.4f")
        market_price = st.number_input("Option market price", value=10.0000, key="bin_mp", format="%.4f")
        N = st.number_input("Number of steps (N)", value=100, min_value=1, step=1, key="bin_n")
        if st.button("Calculate Theoretical Price (Binomial)", key="bin_btn"):
            try:
                iv = implied_volatility_option(market_price, S, K, T, r, option_type, max_iter=100)
                price = binomial_european_option_price(S, K, T, r, iv, int(N), option_type)
                greeks = binomial_greeks_european_option(S, K, T, r, iv, int(N), option_type)
                st.success(f"Binomial theoretical price ({option_type}): {price:.4f} (using IV: {iv:.4f})")
                st.write("Greeks:")
                st.json(greeks)
            except Exception as e:
                st.error(f"Error in Binomial calculation: {e}")

elif menu == "Monte Carlo":
    import importlib.util
    import sys
    import os
    monte_carlo_path = os.path.join(os.path.dirname(__file__), 'monte-carlo', 'european_options')
    call_path = os.path.join(monte_carlo_path, 'call.py')
    put_path = os.path.join(monte_carlo_path, 'put.py')
    try:
        spec_call = importlib.util.spec_from_file_location('mc_call', call_path)
        mc_call = importlib.util.module_from_spec(spec_call)
        spec_call.loader.exec_module(mc_call)
        spec_put = importlib.util.spec_from_file_location('mc_put', put_path)
        mc_put = importlib.util.module_from_spec(spec_put)
        spec_put.loader.exec_module(mc_put)
        implied_vol_call = mc_call.implied_volatility_newton
        implied_vol_put = mc_put.implied_volatility_newton_put
        monte_carlo_european_call = mc_call.monte_carlo_european_call
        monte_carlo_greeks_call = mc_call.monte_carlo_greeks_call
        monte_carlo_european_put = mc_put.monte_carlo_european_put
        monte_carlo_greeks_put = mc_put.monte_carlo_greeks_put
    except Exception as e:
        st.error(f"Error importing the Monte Carlo model: {e}")
        st.stop()
    st.write("Calculate the theoretical price and Greeks of a European option using Monte Carlo simulation.")
    option_type = st.selectbox("Option type", ["call", "put"], key="mc_type")
    S = st.number_input("Spot price (S)", value=100.0000, key="mc_s", format="%.4f")
    K = st.number_input("Strike price (K)", value=100.0000, key="mc_k", format="%.4f")
    T = st.number_input("Time to maturity (years)", value=1.0000, key="mc_t", format="%.4f")
    r = st.number_input("Risk-free rate (r, decimal)", value=0.0500, key="mc_r", format="%.4f")
    market_price = st.number_input("Option market price", value=10.0000, key="mc_mp", format="%.4f")
    n_sim = st.number_input("Number of Monte Carlo simulations", value=10000, min_value=1000, step=1000, key="mc_nsim")
    if st.button("Calculate Theoretical Price (Monte Carlo)", key="mc_btn"):
        try:
            if option_type == "call":
                iv = implied_vol_call(market_price, S, K, T, r)
            else:
                iv = implied_vol_put(market_price, S, K, T, r)
            if iv is None:
                st.error("Implied volatility could not be determined for the given parameters. Please check your inputs.")
            else:
                if option_type == "call":
                    price = monte_carlo_european_call(S, K, T, r, iv, int(n_sim))
                    greeks = monte_carlo_greeks_call(S, K, T, r, iv, int(n_sim))
                else:
                    price = monte_carlo_european_put(S, K, T, r, iv, int(n_sim))
                    greeks = monte_carlo_greeks_put(S, K, T, r, iv, int(n_sim))
                st.success(f"Monte Carlo theoretical price ({option_type}): {price:.4f} (using IV: {iv:.4f})")
                st.write("Greeks:")
                st.json(greeks)
        except Exception as e:
            st.error(f"Error in Monte Carlo calculation: {e}")

elif menu == "Finite Differences":
    try:
        from finite_difference_method.european_options.call import (
            implied_volatility_call, finite_difference_european_call, finite_difference_greeks_call
        )
        from finite_difference_method.european_options.put import (
            implied_volatility_put, finite_difference_european_put, finite_difference_greeks_put
        )
    except Exception as e:
        st.error(f"Error importing the Finite Differences method: {e}")
    else:
        st.write("Calculate the theoretical price and Greeks of a European option using the Finite Differences method.")
        option_type = st.selectbox("Option type", ["call", "put"], key="fd_type")
        S = st.number_input("Spot price (S)", value=100.0000, key="fd_s", format="%.4f")
        K = st.number_input("Strike price (K)", value=100.0000, key="fd_k", format="%.4f")
        T = st.number_input("Time to maturity (years)", value=1.0000, key="fd_t", format="%.4f")
        r = st.number_input("Risk-free rate (r, decimal)", value=0.0500, key="fd_r", format="%.4f")
        market_price = st.number_input("Option market price", value=10.0000, key="fd_mp", format="%.4f")
        M = st.number_input("Number of price steps (M)", value=100, min_value=10, step=10, key="fd_m")
        N = st.number_input("Number of time steps (N)", value=100, min_value=10, step=10, key="fd_n")
        Smax = st.number_input("Smax (multiple of K)", value=2.0000, min_value=1.0, step=0.1, key="fd_smax", format="%.4f")
        if st.button("Calculate Theoretical Price (Finite Differences)", key="fd_btn"):
            try:
                if option_type == "call":
                    iv = implied_volatility_call(market_price, S, K, T, r)
                    price = finite_difference_european_call(S, K, T, r, iv, Smax, int(M), int(N))
                    greeks = finite_difference_greeks_call(S, K, T, r, iv, Smax, int(M), int(N))
                else:
                    iv = implied_volatility_put(market_price, S, K, T, r)
                    price = finite_difference_european_put(S, K, T, r, iv, Smax, int(M), int(N))
                    greeks = finite_difference_greeks_put(S, K, T, r, iv, Smax, int(M), int(N))
                st.success(f"Finite Differences theoretical price ({option_type}): {price:.4f} (using IV: {iv:.4f})")
                st.write("Greeks:")
                st.json(greeks)
            except Exception as e:
                st.error(f"Error in Finite Differences calculation: {e}")

elif menu == "Model Comparison":
    import importlib.util
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    st.write("### Model Comparison")
    st.write("Compare the theoretical prices of your favorite models side by side.")
    option_type = st.selectbox("Option type", ["call", "put"], key="cmp_type")
    S = st.number_input("Spot price (S)", value=100.0000, key="cmp_s", format="%.4f")
    K = st.number_input("Strike price (K)", value=100.0000, key="cmp_k", format="%.4f")
    T = st.number_input("Time to maturity (years)", value=1.0000, key="cmp_t", format="%.4f")
    r = st.number_input("Risk-free rate (r, decimal)", value=0.0500, key="cmp_r", format="%.4f")
    market_price = st.number_input("Option market price", value=10.0000, key="cmp_mp", format="%.4f")
    N_bin = st.number_input("Number of Binomial steps (N)", value=100, min_value=1, step=1, key="cmp_nbin")
    n_sim = st.number_input("Number of Monte Carlo simulations", value=10000, min_value=1000, step=1000, key="cmp_nsim")
    M_fd = st.number_input("Number of price steps (M) FD", value=100, min_value=10, step=10, key="cmp_mfd")
    N_fd = st.number_input("Number of time steps (N) FD", value=100, min_value=10, step=10, key="cmp_nfd")
    Smax = st.number_input("Smax (multiple of K) FD", value=2.0000, min_value=1.0, step=0.1, key="cmp_smax", format="%.4f")
    if st.button("Compare Models", key="cmp_btn"):
        try:
            # Binomial
            from binomial_model.european_options.binomial import implied_volatility_option, binomial_european_option_price
            iv = implied_volatility_option(market_price, S, K, T, r, option_type, max_iter=100)
            price_bin = binomial_european_option_price(S, K, T, r, iv, int(N_bin), option_type)
            # Monte Carlo
            monte_carlo_path = os.path.join(os.path.dirname(__file__), 'monte-carlo', 'european_options')
            call_path = os.path.join(monte_carlo_path, 'call.py')
            put_path = os.path.join(monte_carlo_path, 'put.py')
            spec_call = importlib.util.spec_from_file_location('mc_call', call_path)
            mc_call = importlib.util.module_from_spec(spec_call)
            spec_call.loader.exec_module(mc_call)
            spec_put = importlib.util.spec_from_file_location('mc_put', put_path)
            mc_put = importlib.util.module_from_spec(spec_put)
            spec_put.loader.exec_module(mc_put)
            if option_type == "call":
                price_mc = mc_call.monte_carlo_european_call(S, K, T, r, iv, int(n_sim))
            else:
                price_mc = mc_put.monte_carlo_european_put(S, K, T, r, iv, int(n_sim))
            # Finite Differences
            if option_type == "call":
                from finite_difference_method.european_options.call import finite_difference_european_call
                price_fd = finite_difference_european_call(S, K, T, r, iv, Smax, int(M_fd), int(N_fd))
            else:
                from finite_difference_method.european_options.put import finite_difference_european_put
                price_fd = finite_difference_european_put(S, K, T, r, iv, Smax, int(M_fd), int(N_fd))
            # Visualization
            labels = ["Binomial", "Monte Carlo", "Finite Differences", "Market"]
            values = [price_bin, price_mc, price_fd, market_price]
            fig, ax = plt.subplots()
            bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
            ax.set_ylabel("Option Price")
            ax.set_title(f"Model Comparison (IV={iv:.4f})")
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", va='bottom', ha='center', fontsize=10)
            st.pyplot(fig)
            st.write("#### Numerical Results:")
            st.write({
                "Binomial Price": price_bin,
                "Monte Carlo Price": price_mc,
                "Finite Differences Price": price_fd,
                "Market Price": market_price,
                "Implied Volatility used": iv
            })
        except Exception as e:
            st.error(f"Error in model comparison: {e}")

st.markdown('<div class="footer-conference">Developed by Marcos Heredia Pimienta, Quantitative Risk Analyst', unsafe_allow_html=True)
