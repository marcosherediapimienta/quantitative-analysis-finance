import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Option Pricing & Portfolio Risk App", layout="centered")

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
        "Single Option Analysis",
        "Portfolio Analysis",
        "Model Comparison",
        "Visualizations",
        "Hedging & Sensitivity Analysis"
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
                    from option_pricing.black_scholes_model import bs_portfolio_analysis as bsa
                    if option_style == "European":
                        iv = bsa.implied_volatility_option(market_price, S, K, T, r, option_type)
                        price = bsa.price_option_bs(S, K, T, r, iv, option_type)
                        greeks = bsa.greeks_bs(S, K, T, r, iv, option_type)
                    else:
                        st.info("Black-Scholes is not suitable for American options. Use Binomial or Monte Carlo.")
                        price = None
                        greeks = None
                elif model == "Binomial":
                    from option_pricing.binomial_model import binomial_portfolio_analysis as bpa
                    if option_style == "European":
                        iv = bpa.implied_volatility_option(market_price, S, K, T, r, option_type)
                        price = bpa.binomial_european_option_price(S, K, T, r, iv, int(N), option_type)
                        greeks = bpa.binomial_greeks_european_option(S, K, T, r, iv, int(N), option_type)
                    else:
                        price = bpa.binomial_american_option_price(S, K, T, r, market_price, int(N), option_type)
                        greeks = bpa.binomial_greeks_american_option(S, K, T, r, market_price, int(N), option_type)
                        iv = None
                elif model == "Monte Carlo":
                    from option_pricing.monte_carlo import mc_portfolio_analysis as mca
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

if menu == "Portfolio Analysis":
    st.header("Portfolio Analysis")
    st.write("Define your portfolio of options (European or American, any model) and analyze risk metrics and visualizations.")
    st.info("For advanced risk analysis, use the Binomial or Monte Carlo models.")
    
    n_opts = st.number_input("Number of options in portfolio", min_value=1, max_value=10, value=st.session_state.get("n_opts", 3), key="n_opts", help="How many different options do you want to include in your portfolio?")
    portfolio = []
    for i in range(int(n_opts)):
        st.subheader(f"Option #{i+1}")
        cols = st.columns(3)
        with cols[0]:
            model = st.selectbox(f"Model", ["Binomial", "Monte Carlo"], key=f"model_{i}", help="Pricing model for this option.")
            style = st.selectbox(f"Style", ["European", "American"], key=f"style_{i}", help="Option exercise style.")
            typ = st.selectbox(f"Type", ["call", "put"], key=f"type_{i}", help="Call or put option.")
        with cols[1]:
            S = st.number_input(f"Spot", value=st.session_state.get(f"S_{i}", 100.0), key=f"S_{i}", help="Current price of the underlying asset.")
            K = st.number_input(f"Strike", value=st.session_state.get(f"K_{i}", 100.0), key=f"K_{i}", help="Strike price of the option.")
            T = st.number_input(f"Maturity (years)", value=st.session_state.get(f"T_{i}", 1.0), key=f"T_{i}", min_value=0.01, help="Time to maturity in years.")
        with cols[2]:
            r = st.number_input(f"Risk-free rate", value=st.session_state.get(f"r_{i}", 0.05), key=f"r_{i}", min_value=0.0, max_value=1.0, step=0.01, help="Annual risk-free interest rate (as decimal, e.g. 0.03 for 3%).")
            qty = st.number_input(f"Quantity", value=st.session_state.get(f"qty_{i}", 1), key=f"qty_{i}", help="Number of contracts (positive: long, negative: short).")
            market_price = st.number_input(f"Market price", value=st.session_state.get(f"mp_{i}", 10.0), key=f"mp_{i}", min_value=0.0, help="Observed market price of the option.")
        # Validación básica
        if T <= 0:
            st.error(f"Option #{i+1}: Maturity must be positive.")
        if qty == 0:
            st.warning(f"Option #{i+1}: Quantity is zero, this option will not affect the portfolio.")
        portfolio.append({'type': typ, 'style': style.lower(), 'S': S, 'K': K, 'T': T, 'r': r, 'qty': qty, 'market_price': market_price})
    N = st.number_input("Number of steps (Binomial/MC)", value=100, min_value=1, step=1, key="port_n", help="Discretization steps for Binomial/Monte Carlo models.")
    n_sim = st.number_input("Number of Monte Carlo simulations", value=1000, min_value=100, step=100, key="port_nsim", help="Number of scenarios for risk simulation.")
    horizon = st.number_input("Risk horizon (days)", value=10, min_value=1, step=1, key="port_horizon", help="Number of days for VaR/ES calculation.") / 252
    if st.button("Analyze Portfolio", key="analyze_portfolio_btn"):
        with st.spinner("Calculating risk metrics and simulating portfolio..."):
            try:
                from option_pricing.binomial_model import binomial_portfolio_analysis as bpa
                from option_pricing.monte_carlo import mc_portfolio_analysis as mca
                model_set = set([st.session_state[f"model_{i}"] for i in range(int(n_opts))])
                if len(model_set) > 1:
                    st.warning("For now, all options in the portfolio must use the same model (Binomial or Monte Carlo).")
                else:
                    model = model_set.pop()
                    if model == "Binomial":
                        sim = bpa.simulate_portfolio(portfolio, n_sims=int(n_sim), N=int(N), horizon=horizon)
                        pnl = sim['pnl']
                        var, es = bpa.var_es(pnl, alpha=0.01)
                        value = bpa.portfolio_value(portfolio, N=int(N))
                        greeks = bpa.portfolio_greeks(portfolio, N=int(N))
                    else:
                        sim = mca.simulate_portfolio_mc_pricing(portfolio, n_sims=int(n_sim), n_steps=int(N), horizon=horizon)
                        pnl = sim['pnl']
                        var, es = mca.var_es(pnl, alpha=0.01)
                        value = sum(mca.price_option_mc(opt, n_sim=10000, n_steps=int(N)) * opt['qty'] for opt in portfolio)
                        greeks = mca.portfolio_greeks_mc(portfolio, n_sim=10000, n_steps=int(N))
                    # Guardar resultados en session_state
                    st.session_state['portfolio_results'] = {
                        'model': model,
                        'portfolio': portfolio,
                        'pnl': pnl,
                        'var': var,
                        'es': es,
                        'value': value,
                        'greeks': greeks,
                        'sim': sim,
                        'N': int(N),
                        'n_sim': int(n_sim),
                        'horizon': horizon
                    }
                    # Resultados destacados
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Portfolio Value", f"{value:.2f}")
                    col2.metric("VaR (99%)", f"{var:.2f}")
                    col3.metric("ES (99%)", f"{es:.2f}")
                    st.markdown("---")
                    st.write("Greeks (aggregated):")
                    st.json(greeks)
                    # Mostrar histograma
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.hist(pnl, bins=50, color='skyblue', edgecolor='k', alpha=0.7, density=True)
                    ax.axvline(-var, color='red', linestyle='--', label=f'VaR (99%)')
                    ax.axvline(-es, color='orange', linestyle='--', label=f'ES (99%)')
                    ax.set_title('Simulated P&L distribution of the portfolio')
                    ax.set_xlabel('P&L')
                    ax.set_ylabel('Density')
                    ax.legend()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in portfolio analysis: {e}")

if menu == "Model Comparison":
    st.header("Model Comparison")
    st.write("Compare the price and Greeks of a single option across all models.")
    cols = st.columns(3)
    with cols[0]:
        option_style = st.selectbox("Option style", ["European", "American"], key="cmp_style", help="Exercise style of the option.")
        option_type = st.selectbox("Option type", ["call", "put"], key="cmp_type", help="Call or put option.")
    with cols[1]:
        S = st.number_input("Spot price (S)", value=100.0, key="cmp_s", format="%.4f", help="Current price of the underlying asset.")
        K = st.number_input("Strike price (K)", value=100.0, key="cmp_k", format="%.4f", help="Strike price of the option.")
        T = st.number_input("Time to maturity (years)", value=1.0, key="cmp_t", format="%.4f", min_value=0.01, help="Time to maturity in years.")
    with cols[2]:
        r = st.number_input("Risk-free rate (r, decimal)", value=0.05, key="cmp_r", format="%.4f", min_value=0.0, max_value=1.0, step=0.01, help="Annual risk-free interest rate (as decimal, e.g. 0.03 for 3%).")
        market_price = st.number_input("Option market price", value=10.0, key="cmp_mp", format="%.4f", min_value=0.0, help="Observed market price of the option.")
        N = st.number_input("Number of steps (Binomial/MC)", value=100, min_value=1, step=1, key="cmp_n", help="Discretization steps for Binomial/Monte Carlo models.")
        n_sim = st.number_input("Number of Monte Carlo simulations", value=10000, min_value=1000, step=1000, key="cmp_nsim", help="Number of scenarios for risk simulation.")
    if T <= 0:
        st.error("Maturity must be positive.")
    if st.button("Compare Models", key="cmp_btn"):
        with st.spinner("Comparing models..."):
            try:
                from option_pricing.black_scholes_model import bs_portfolio_analysis as bsa
                from option_pricing.binomial_model import binomial_portfolio_analysis as bpa
                from option_pricing.monte_carlo import mc_portfolio_analysis as mca
                results = {}
                if option_style == "European":
                    iv_bs = bsa.implied_volatility_option(market_price, S, K, T, r, option_type)
                    price_bs = bsa.price_option_bs(S, K, T, r, iv_bs, option_type)
                    greeks_bs = bsa.greeks_bs(S, K, T, r, iv_bs, option_type)
                    iv_bin = bpa.implied_volatility_option(market_price, S, K, T, r, option_type)
                    price_bin = bpa.binomial_european_option_price(S, K, T, r, iv_bin, int(N), option_type)
                    greeks_bin = bpa.binomial_greeks_european_option(S, K, T, r, iv_bin, int(N), option_type)
                    opt = {'type': option_type, 'style': 'european', 'S': S, 'K': K, 'T': T, 'r': r, 'qty': 1, 'market_price': market_price}
                    price_mc = mca.price_option_mc(opt, n_sim=int(n_sim), n_steps=int(N))
                    greeks_mc = mca.option_greeks_mc(opt, n_sim=int(n_sim), n_steps=int(N))
                    results['Black-Scholes'] = {'price': price_bs, 'iv': iv_bs, 'greeks': greeks_bs}
                    results['Binomial'] = {'price': price_bin, 'iv': iv_bin, 'greeks': greeks_bin}
                    results['Monte Carlo'] = {'price': price_mc, 'iv': None, 'greeks': greeks_mc}
                else:
                    price_bin = bpa.binomial_american_option_price(S, K, T, r, market_price, int(N), option_type)
                    greeks_bin = bpa.binomial_greeks_american_option(S, K, T, r, market_price, int(N), option_type)
                    opt = {'type': option_type, 'style': 'american', 'S': S, 'K': K, 'T': T, 'r': r, 'qty': 1, 'market_price': market_price}
                    price_mc = mca.price_option_mc(opt, n_sim=int(n_sim), n_steps=int(N))
                    greeks_mc = mca.option_greeks_mc(opt, n_sim=int(n_sim), n_steps=int(N))
                    results['Binomial'] = {'price': price_bin, 'iv': None, 'greeks': greeks_bin}
                    results['Monte Carlo'] = {'price': price_mc, 'iv': None, 'greeks': greeks_mc}
                # Resultados destacados
                st.markdown("---")
                for model, res in results.items():
                    st.subheader(model)
                    col1, col2 = st.columns(2)
                    col1.metric("Model Price", f"{res['price']:.4f}")
                    if res['iv'] is not None:
                        col2.metric("Implied Volatility", f"{res['iv']:.4f}")
                    st.write("Greeks:")
                    st.json(res['greeks'])
            except Exception as e:
                st.error(f"Error in model comparison: {e}")

if menu == "Visualizations":
    st.header("Visualizations")
    st.write("Browse generated histograms and sensitivity plots from your analyses.")
    vis_dir = os.path.join(os.path.dirname(__file__), 'option_pricing', 'visualizations')
    if not os.path.exists(vis_dir):
        st.warning("No visualizations found yet. Run some analyses first.")
    else:
        files = [f for f in os.listdir(vis_dir) if f.endswith('.png')]
        if not files:
            st.warning("No PNG files found in visualizations directory.")
        else:
            # Mostrar previews en grid
            n_cols = 3
            rows = [files[i:i+n_cols] for i in range(0, len(files), n_cols)]
            st.markdown("#### Previews:")
            for row in rows:
                cols = st.columns(n_cols)
                for idx, file in enumerate(row):
                    file_path = os.path.join(vis_dir, file)
                    with open(file_path, "rb") as img_file:
                        img_bytes = img_file.read()
                    cols[idx].image(img_bytes, caption=file, use_container_width=True)
                    cols[idx].download_button(
                        label="Download",
                        data=img_bytes,
                        file_name=file,
                        mime="image/png",
                        help=f"Download {file} ({os.path.getsize(file_path)//1024} KB)"
                    )
            st.markdown("---")
            # Visualización ampliada
            file = st.selectbox("Select a visualization to display in large:", files)
            file_path = os.path.join(vis_dir, file)
            st.image(file_path, use_container_width=True, caption=f"{file} ({os.path.getsize(file_path)//1024} KB)")
            with open(file_path, "rb") as img_file:
                st.download_button(
                    label="Download selected image",
                    data=img_file.read(),
                    file_name=file,
                    mime="image/png",
                    help=f"Download {file} ({os.path.getsize(file_path)//1024} KB)"
                )

if menu == "Hedging & Sensitivity Analysis":
    st.header("Hedging & Sensitivity Analysis")
    st.write("Select the type of hedge and analyze the impact on your portfolio.")
    hedge_type = st.selectbox(
        "Select hedge type:",
        ["Delta", "DeltaGamma", "Vega"],
        index=0,
        help="Choose the type of hedge to analyze: Delta, Delta+Gamma, or Vega."
    )
    if 'portfolio_results' in st.session_state:
        results = st.session_state['portfolio_results']
        st.success(f"Loaded portfolio results for model: {results['model']}")
        st.write("**Portfolio Value:**", results['value'])
        st.write("**VaR (99%):**", results['var'])
        st.write("**ES (99%):**", results['es'])
        st.write("**Greeks (aggregated):**")
        st.json(results['greeks'])
        st.write("---")
        # Definir vis_dir para ambos modelos
        vis_dir = os.path.join(os.path.dirname(__file__), 'option_pricing', 'visualizations')
        if results['model'] == 'Binomial':
            from option_pricing.binomial_model import binomial_portfolio_analysis as bpa
            import copy
            import matplotlib.pyplot as plt
            portfolio = results['portfolio']
            N = results['N']
            horizon = results['horizon']
            n_sims = results['n_sim']
            # Botón para generar archivos de sensibilidad Binomial
            if st.button("Generar/Actualizar análisis de sensibilidad (Binomial)"):
                with st.spinner("Generando análisis de sensibilidad Binomial..."):
                    bpa.run_sensitivity_analysis_binomial(
                        portfolio=portfolio,
                        N=N,
                        vis_dir=vis_dir
                    )
                st.success("¡Archivos de sensibilidad Binomial generados!")
                st.rerun()
            sim = bpa.simulate_portfolio(portfolio, n_sims=n_sims, N=N, horizon=horizon)
            pnl = sim['pnl']
            shocks = sim['shocks']
            value = bpa.portfolio_value(portfolio, N=N)
            var, es = bpa.var_es(pnl, alpha=0.01)
            greeks_total = bpa.portfolio_greeks(portfolio, N=N)
            # Delta
            if hedge_type == 'Delta':
                subyacentes = {}
                for opt in portfolio:
                    key = opt.get('ticker', opt['S'])
                    greeks = bpa.option_greeks(opt, N)
                    subyacentes.setdefault(key, {'S0': opt['S'], 'delta': 0})
                    subyacentes[key]['delta'] += greeks['delta'] * opt['qty']
                pnl_hedged = []
                for i in range(len(pnl)):
                    hedge_pnl = 0
                    for key, v in subyacentes.items():
                        S0 = v['S0']
                        delta = v['delta']
                        Z = shocks[key][i]
                        for opt in portfolio:
                            if opt.get('ticker', opt['S']) == key:
                                from option_pricing.binomial_model.binomial_portfolio_analysis import implied_volatility_option
                                iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                if iv is None:
                                    iv = 0.2
                                r = opt['r']
                                T_sim = horizon if horizon is not None else opt['T']
                                break
                        S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                        hedge_pnl += -delta * (S_T - S0)
                    pnl_hedged.append(pnl[i] + hedge_pnl)
                import numpy as np
                pnl_hedged = np.array(pnl_hedged)
                var_hedged, es_hedged = bpa.var_es(pnl_hedged, alpha=0.01)
                st.subheader("Delta Hedge Results")
                st.write(f"VaR after delta hedge (99%): {var_hedged:.2f}")
                st.write(f"ES after delta hedge (99%): {es_hedged:.2f}")
                st.write(f"VaR reduction: {var - var_hedged:.2f}")
                st.write(f"ES reduction: {es - es_hedged:.2f}")
                fig, ax = plt.subplots(figsize=(8,4))
                ax.hist(pnl, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                ax.hist(pnl_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
                ax.axvline(-var, color='blue', linestyle='--', label=f'VaR Original ({-var:.2f})')
                ax.axvline(-es, color='blue', linestyle=':', label=f'ES Original ({-es:.2f})')
                ax.axvline(-var_hedged, color='orange', linestyle='--', label=f'VaR Delta ({-var_hedged:.2f})')
                ax.axvline(-es_hedged, color='orange', linestyle=':', label=f'ES Delta ({-es_hedged:.2f})')
                ax.set_title('P&L Distribution: Delta Hedge (Binomial)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            # DeltaGamma
            elif hedge_type == 'DeltaGamma':
                greeks_total = bpa.portfolio_greeks(portfolio, N)
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
                greeks_hedge = bpa.option_greeks(hedge_opt, N)
                gamma_hedge = greeks_hedge['gamma']
                gamma_hedge_fraction = 0.7
                qty_gamma_hedge = -gamma_cartera * gamma_hedge_fraction / gamma_hedge if gamma_hedge != 0 else 0
                hedge_opt['qty'] = qty_gamma_hedge
                portfolio_gamma_hedged = portfolio + [hedge_opt]
                greeks_total_gamma = bpa.portfolio_greeks(portfolio_gamma_hedged, N)
                delta_gamma_hedged = greeks_total_gamma['delta']
                pnl_gamma_delta_hedged = []
                for i in range(len(pnl)):
                    shocked_portfolio = []
                    for opt in portfolio_gamma_hedged:
                        key = opt.get('ticker', opt['S'])
                        S0 = opt['S']
                        from option_pricing.binomial_model.binomial_portfolio_analysis import implied_volatility_option
                        iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                        if iv is None:
                            iv = 0.2
                        r = opt['r']
                        T_sim = horizon if horizon is not None else opt['T']
                        Z = shocks[key][i] if key in shocks else np.random.normal(0, 1)
                        S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                        shocked_opt = opt.copy()
                        shocked_opt['S'] = S_T
                        shocked_portfolio.append(shocked_opt)
                    val = bpa.portfolio_value(shocked_portfolio, N=N)
                    hedge_pnl = 0
                    S0 = portfolio[0]['S']
                    delta = delta_gamma_hedged
                    Z = shocks[portfolio[0].get('ticker', portfolio[0]['S'])][i]
                    S_T = S0 * np.exp((portfolio[0]['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                    hedge_pnl += -delta * (S_T - S0)
                    pnl_gamma_delta_hedged.append(val - value + hedge_pnl)
                import numpy as np
                pnl_gamma_delta_hedged = np.array(pnl_gamma_delta_hedged)
                var_gamma_delta_hedged, es_gamma_delta_hedged = bpa.var_es(pnl_gamma_delta_hedged, alpha=0.01)
                st.subheader("Gamma+Delta Hedge Results")
                st.write(f"VaR after gamma+delta hedge (99%): {var_gamma_delta_hedged:.2f}")
                st.write(f"ES after gamma+delta hedge (99%): {es_gamma_delta_hedged:.2f}")
                st.write(f"VaR reduction vs original: {var - var_gamma_delta_hedged:.2f}")
                st.write(f"ES reduction vs original: {es - es_gamma_delta_hedged:.2f}")
                fig, ax = plt.subplots(figsize=(8,4))
                ax.hist(pnl, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                ax.hist(pnl_gamma_delta_hedged, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
                ax.axvline(-var, color='blue', linestyle='--', label=f'VaR Original ({-var:.2f})')
                ax.axvline(-es, color='blue', linestyle=':', label=f'ES Original ({-es:.2f})')
                ax.axvline(-var_gamma_delta_hedged, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_gamma_delta_hedged:.2f})')
                ax.axvline(-es_gamma_delta_hedged, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_gamma_delta_hedged:.2f})')
                ax.set_title('P&L Distribution: Gamma+Delta Hedge (Binomial)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            # Vega
            elif hedge_type == 'Vega':
                vega_total = bpa.portfolio_greeks(portfolio, N)['vega']
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
                greeks_hedge_vega = bpa.option_greeks(hedge_opt_vega, N)
                vega_hedge = greeks_hedge_vega['vega']
                vega_hedge_fraction = 0.7
                qty_vega_hedge = -vega_total * vega_hedge_fraction / vega_hedge if vega_hedge != 0 else 0
                hedge_opt_vega['qty'] = qty_vega_hedge
                portfolio_vega_hedged = portfolio + [hedge_opt_vega]
                pnl_vega_hedged = []
                for i in range(len(pnl)):
                    shocked_portfolio = []
                    for opt in portfolio_vega_hedged:
                        key = opt.get('ticker', opt['S'])
                        S0 = opt['S']
                        from option_pricing.binomial_model.binomial_portfolio_analysis import implied_volatility_option
                        iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                        if iv is None:
                            iv = 0.2
                        r = opt['r']
                        T_sim = horizon if horizon is not None else opt['T']
                        Z = shocks[key][i] if key in shocks else np.random.normal(0, 1)
                        S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                        shocked_opt = opt.copy()
                        shocked_opt['S'] = S_T
                        shocked_portfolio.append(shocked_opt)
                    val = bpa.portfolio_value(shocked_portfolio, N=N)
                    pnl_vega_hedged.append(val - value)
                import numpy as np
                pnl_vega_hedged = np.array(pnl_vega_hedged)
                var_vega_hedged, es_vega_hedged = bpa.var_es(pnl_vega_hedged, alpha=0.01)
                st.subheader("Vega Hedge Results")
                st.write(f"VaR after vega hedge (99%): {var_vega_hedged:.2f}")
                st.write(f"ES after vega hedge (99%): {es_vega_hedged:.2f}")
                st.write(f"VaR reduction vs original: {var - var_vega_hedged:.2f}")
                st.write(f"ES reduction vs original: {es - es_vega_hedged:.2f}")
                fig, ax = plt.subplots(figsize=(8,4))
                ax.hist(pnl, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                ax.hist(pnl_vega_hedged, bins=50, color='purple', edgecolor='k', alpha=0.5, density=True, label='Vega Hedge')
                ax.axvline(-var, color='blue', linestyle='--', label=f'VaR Original ({-var:.2f})')
                ax.axvline(-es, color='blue', linestyle=':', label=f'ES Original ({-es:.2f})')
                ax.axvline(-var_vega_hedged, color='purple', linestyle='--', label=f'VaR Vega ({-var_vega_hedged:.2f})')
                ax.axvline(-es_vega_hedged, color='purple', linestyle=':', label=f'ES Vega ({-es_vega_hedged:.2f})')
                ax.set_title('P&L Distribution: Vega Hedge (Binomial)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            # Mostrar análisis de sensibilidad si existen los archivos
            sens_files = [
                ('Spot Sensitivity', 'sensitivity_spot_all.png'),
                ('r Sensitivity', 'sensitivity_r_all.png'),
                ('Volatility Sensitivity', 'sensitivity_vol_all.png'),
            ]
            st.markdown('---')
            st.subheader('Sensitivity Analysis')
            for title, fname in sens_files:
                file_path = os.path.join(vis_dir, fname)
                st.markdown(f'**{title}:**')
                if os.path.exists(file_path):
                    st.image(file_path, use_container_width=True)
                else:
                    st.info(f"No {title} plot found. Run the analysis to generate it.")
                # Mostrar datos tabulares si existen (CSV o TXT)
                base_name = fname.replace('.png', '')
                table_shown = False
                for ext in ['.csv', '.txt']:
                    data_path = os.path.join(vis_dir, base_name + ext)
                    if os.path.exists(data_path):
                        try:
                            if ext == '.csv':
                                df = pd.read_csv(data_path)
                            else:
                                df = pd.read_csv(data_path, sep=None, engine='python')
                            # Si la tabla tiene una columna de estrategia, ponla como índice
                            if 'Strategy' in df.columns:
                                df = df.set_index('Strategy')
                            st.dataframe(df, use_container_width=True)
                            # Mostrar también los valores de los ejes si existen en la tabla
                            st.markdown('**Raw sensitivity values:**')
                            st.write(df.to_dict(orient='list'))
                            table_shown = True
                        except Exception as e:
                            st.warning(f"Could not display table for {title}: {e}")
                # Si no hay archivo, intenta mostrar una tabla resumen si los datos están en memoria (placeholder)
                if not table_shown:
                    st.info(f"No summary table found for {title}. Run the analysis to generate it.")
        elif results['model'] == 'Monte Carlo':
            from option_pricing.monte_carlo import mc_portfolio_analysis as mca
            import copy
            import matplotlib.pyplot as plt
            import numpy as np
            portfolio = results['portfolio']
            N = results['N']
            horizon = results['horizon']
            n_sims = results['n_sim']
            # Botón para generar archivos de sensibilidad Monte Carlo
            if st.button("Generar/Actualizar análisis de sensibilidad (Monte Carlo)"):
                with st.spinner("Generando análisis de sensibilidad Monte Carlo..."):
                    mca.run_sensitivity_analysis_mc(
                        portfolio=portfolio,
                        N=N,
                        n_sim_sens=20000,
                        vis_dir=vis_dir,
                        horizon=horizon
                    )
                st.success("¡Archivos de sensibilidad Monte Carlo generados!")
                st.rerun()
            sim = mca.simulate_portfolio_mc_pricing(portfolio, n_sims=n_sims, n_steps=N, horizon=horizon)
            pnl = sim['pnl']
            shocks = sim['shocks']
            value = sum(mca.price_option_mc(opt, n_sim=10000, n_steps=N) * opt['qty'] for opt in portfolio)
            var, es = mca.var_es(pnl, alpha=0.01)
            greeks_total = mca.portfolio_greeks_mc(portfolio, n_sim=10000, n_steps=N)
            # Delta
            if hedge_type == 'Delta':
                subyacentes = {}
                for opt in portfolio:
                    key = opt.get('ticker', opt['S'])
                    greeks = mca.option_greeks_mc(opt, n_sim=10000, n_steps=N)
                    subyacentes.setdefault(key, {'S0': opt['S'], 'delta': 0})
                    subyacentes[key]['delta'] += greeks['delta'] * opt['qty']
                pnl_hedged = []
                for i in range(len(pnl)):
                    hedge_pnl = 0
                    for key, v in subyacentes.items():
                        S0 = v['S0']
                        delta = v['delta']
                        Z = shocks[key][i]
                        for opt in portfolio:
                            if opt.get('ticker', opt['S']) == key:
                                from option_pricing.monte_carlo.mc_portfolio_analysis import implied_volatility_option
                                iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                                if iv is None:
                                    iv = 0.2
                                r = opt['r']
                                T_sim = horizon if horizon is not None else opt['T']
                                break
                        S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                        hedge_pnl += -delta * (S_T - S0)
                    pnl_hedged.append(pnl[i] + hedge_pnl)
                pnl_hedged = np.array(pnl_hedged)
                var_hedged, es_hedged = mca.var_es(pnl_hedged, alpha=0.01)
                st.subheader("Delta Hedge Results (Monte Carlo)")
                st.write(f"VaR after delta hedge (99%): {var_hedged:.2f}")
                st.write(f"ES after delta hedge (99%): {es_hedged:.2f}")
                st.write(f"VaR reduction: {var - var_hedged:.2f}")
                st.write(f"ES reduction: {es - es_hedged:.2f}")
                fig, ax = plt.subplots(figsize=(8,4))
                ax.hist(pnl, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                ax.hist(pnl_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
                ax.axvline(-var, color='blue', linestyle='--', label=f'VaR Original ({-var:.2f})')
                ax.axvline(-es, color='blue', linestyle=':', label=f'ES Original ({-es:.2f})')
                ax.axvline(-var_hedged, color='orange', linestyle='--', label=f'VaR Delta ({-var_hedged:.2f})')
                ax.axvline(-es_hedged, color='orange', linestyle=':', label=f'ES Delta ({-es_hedged:.2f})')
                ax.set_title('P&L Distribution: Delta Hedge (Monte Carlo)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            # DeltaGamma
            elif hedge_type == 'DeltaGamma':
                greeks_total = mca.portfolio_greeks_mc(portfolio, n_sim=10000, n_steps=N)
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
                greeks_hedge = mca.option_greeks_mc(hedge_opt, n_sim=10000, n_steps=N)
                gamma_hedge = greeks_hedge['gamma']
                gamma_hedge_fraction = 0.7
                qty_gamma_hedge = -gamma_cartera * gamma_hedge_fraction / gamma_hedge if gamma_hedge != 0 else 0
                hedge_opt['qty'] = qty_gamma_hedge
                portfolio_gamma_hedged = portfolio + [hedge_opt]
                greeks_total_gamma = mca.portfolio_greeks_mc(portfolio_gamma_hedged, n_sim=10000, n_steps=N)
                delta_gamma_hedged = greeks_total_gamma['delta']
                pnl_gamma_delta_hedged = []
                for i in range(len(pnl)):
                    shocked_portfolio = []
                    for opt in portfolio_gamma_hedged:
                        key = opt.get('ticker', opt['S'])
                        S0 = opt['S']
                        from option_pricing.monte_carlo.mc_portfolio_analysis import implied_volatility_option
                        iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                        if iv is None:
                            iv = 0.2
                        r = opt['r']
                        T_sim = horizon if horizon is not None else opt['T']
                        Z = shocks[key][i] if key in shocks else np.random.normal(0, 1)
                        S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                        shocked_opt = opt.copy()
                        shocked_opt['S'] = S_T
                        shocked_portfolio.append(shocked_opt)
                    val = sum(mca.price_option_mc(opt, n_sim=500, n_steps=N) * opt['qty'] for opt in shocked_portfolio)
                    hedge_pnl = 0
                    S0 = portfolio[0]['S']
                    delta = delta_gamma_hedged
                    Z = shocks[portfolio[0].get('ticker', portfolio[0]['S'])][i]
                    S_T = S0 * np.exp((portfolio[0]['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                    hedge_pnl += -delta * (S_T - S0)
                    pnl_gamma_delta_hedged.append(val - value + hedge_pnl)
                pnl_gamma_delta_hedged = np.array(pnl_gamma_delta_hedged)
                var_gamma_delta_hedged, es_gamma_delta_hedged = mca.var_es(pnl_gamma_delta_hedged, alpha=0.01)
                st.subheader("Gamma+Delta Hedge Results (Monte Carlo)")
                st.write(f"VaR after gamma+delta hedge (99%): {var_gamma_delta_hedged:.2f}")
                st.write(f"ES after gamma+delta hedge (99%): {es_gamma_delta_hedged:.2f}")
                st.write(f"VaR reduction vs original: {var - var_gamma_delta_hedged:.2f}")
                st.write(f"ES reduction vs original: {es - es_gamma_delta_hedged:.2f}")
                fig, ax = plt.subplots(figsize=(8,4))
                ax.hist(pnl, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                ax.hist(pnl_gamma_delta_hedged, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
                ax.axvline(-var, color='blue', linestyle='--', label=f'VaR Original ({-var:.2f})')
                ax.axvline(-es, color='blue', linestyle=':', label=f'ES Original ({-es:.2f})')
                ax.axvline(-var_gamma_delta_hedged, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_gamma_delta_hedged:.2f})')
                ax.axvline(-es_gamma_delta_hedged, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_gamma_delta_hedged:.2f})')
                ax.set_title('P&L Distribution: Gamma+Delta Hedge (Monte Carlo)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            # Vega
            elif hedge_type == 'Vega':
                vega_total = mca.portfolio_greeks_mc(portfolio, n_sim=10000, n_steps=N)['vega']
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
                greeks_hedge_vega = mca.option_greeks_mc(hedge_opt_vega, n_sim=10000, n_steps=N)
                vega_hedge = greeks_hedge_vega['vega']
                vega_hedge_fraction = 0.7
                qty_vega_hedge = -vega_total * vega_hedge_fraction / vega_hedge if vega_hedge != 0 else 0
                hedge_opt_vega['qty'] = qty_vega_hedge
                portfolio_vega_hedged = portfolio + [hedge_opt_vega]
                pnl_vega_hedged = []
                for i in range(len(pnl)):
                    shocked_portfolio = []
                    for opt in portfolio_vega_hedged:
                        key = opt.get('ticker', opt['S'])
                        S0 = opt['S']
                        from option_pricing.monte_carlo.mc_portfolio_analysis import implied_volatility_option
                        iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                        if iv is None:
                            iv = 0.2
                        r = opt['r']
                        T_sim = horizon if horizon is not None else opt['T']
                        Z = shocks[key][i] if key in shocks else np.random.normal(0, 1)
                        S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
                        shocked_opt = opt.copy()
                        shocked_opt['S'] = S_T
                        shocked_portfolio.append(shocked_opt)
                    val = sum(mca.price_option_mc(opt, n_sim=500, n_steps=N) * opt['qty'] for opt in shocked_portfolio)
                    pnl_vega_hedged.append(val - value)
                pnl_vega_hedged = np.array(pnl_vega_hedged)
                var_vega_hedged, es_vega_hedged = mca.var_es(pnl_vega_hedged, alpha=0.01)
                st.subheader("Vega Hedge Results (Monte Carlo)")
                st.write(f"VaR after vega hedge (99%): {var_vega_hedged:.2f}")
                st.write(f"ES after vega hedge (99%): {es_vega_hedged:.2f}")
                st.write(f"VaR reduction vs original: {var - var_vega_hedged:.2f}")
                st.write(f"ES reduction vs original: {es - es_vega_hedged:.2f}")
                fig, ax = plt.subplots(figsize=(8,4))
                ax.hist(pnl, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
                ax.hist(pnl_vega_hedged, bins=50, color='purple', edgecolor='k', alpha=0.5, density=True, label='Vega Hedge')
                ax.axvline(-var, color='blue', linestyle='--', label=f'VaR Original ({-var:.2f})')
                ax.axvline(-es, color='blue', linestyle=':', label=f'ES Original ({-es:.2f})')
                ax.axvline(-var_vega_hedged, color='purple', linestyle='--', label=f'VaR Vega ({-var_vega_hedged:.2f})')
                ax.axvline(-es_vega_hedged, color='purple', linestyle=':', label=f'ES Vega ({-es_vega_hedged:.2f})')
                ax.set_title('P&L Distribution: Vega Hedge (Monte Carlo)')
                ax.set_xlabel('P&L')
                ax.set_ylabel('Density')
                ax.legend()
                st.pyplot(fig)
            # Mostrar análisis de sensibilidad si existen los archivos
            sens_files = [
                ('Spot Sensitivity', 'sensitivity_spot_mc.png'),
                ('r Sensitivity', 'sensitivity_r_mc.png'),
                ('Volatility Sensitivity', 'sensitivity_vol_mc.png'),
            ]
            st.markdown('---')
            st.subheader('Sensitivity Analysis')
            for title, fname in sens_files:
                file_path = os.path.join(vis_dir, fname)
                st.markdown(f'**{title}:**')
                if os.path.exists(file_path):
                    st.image(file_path, use_container_width=True)
                else:
                    st.info(f"No {title} plot found. Run the analysis to generate it.")
                # Mostrar datos tabulares si existen (CSV o TXT)
                base_name = fname.replace('.png', '')
                table_shown = False
                for ext in ['.csv', '.txt']:
                    data_path = os.path.join(vis_dir, base_name + ext)
                    if os.path.exists(data_path):
                        try:
                            if ext == '.csv':
                                df = pd.read_csv(data_path)
                            else:
                                df = pd.read_csv(data_path, sep=None, engine='python')
                            # Si la tabla tiene una columna de estrategia, ponla como índice
                            if 'Strategy' in df.columns:
                                df = df.set_index('Strategy')
                            st.dataframe(df, use_container_width=True)
                            # Mostrar también los valores de los ejes si existen en la tabla
                            st.markdown('**Raw sensitivity values:**')
                            st.write(df.to_dict(orient='list'))
                            table_shown = True
                        except Exception as e:
                            st.warning(f"Could not display table for {title}: {e}")
                # Si no hay archivo, intenta mostrar una tabla resumen si los datos están en memoria (placeholder)
                if not table_shown:
                    st.info(f"No summary table found for {title}. Run the analysis to generate it.")
        elif results['model'] == 'Black-Scholes':
            from option_pricing.black_scholes_model import bs_portfolio_analysis as bsa
            import copy
            import matplotlib.pyplot as plt
            portfolio = results['portfolio']
            vis_dir = os.path.join(os.path.dirname(__file__), 'option_pricing', 'visualizations')
            # Botón para generar archivos de sensibilidad Black-Scholes
            if st.button("Generar/Actualizar análisis de sensibilidad (Black-Scholes)"):
                with st.spinner("Generando análisis de sensibilidad Black-Scholes..."):
                    bsa.run_sensitivity_analysis_bs(
                        portfolio=portfolio,
                        vis_dir=vis_dir
                    )
                st.success("¡Archivos de sensibilidad Black-Scholes generados!")
                st.rerun()
            # Mostrar análisis de sensibilidad si existen los archivos
            sens_files = [
                ('Spot Sensitivity', 'sensitivity_spot_all_bs.png'),
                ('r Sensitivity', 'sensitivity_r_all_bs.png'),
                ('Volatility Sensitivity', 'sensitivity_vol_all_bs.png'),
            ]
            st.markdown('---')
            st.subheader('Sensitivity Analysis')
            for title, fname in sens_files:
                file_path = os.path.join(vis_dir, fname)
                st.markdown(f'**{title}:**')
                if os.path.exists(file_path):
                    st.image(file_path, use_container_width=True)
                else:
                    st.info(f"No {title} plot found. Run the analysis to generate it.")
                # Mostrar datos tabulares si existen (CSV o TXT)
                base_name = fname.replace('.png', '')
                table_shown = False
                for ext in ['.csv', '.txt']:
                    data_path = os.path.join(vis_dir, base_name + ext)
                    if os.path.exists(data_path):
                        try:
                            if ext == '.csv':
                                df = pd.read_csv(data_path)
                            else:
                                df = pd.read_csv(data_path, sep=None, engine='python')
                            if 'Strategy' in df.columns:
                                df = df.set_index('Strategy')
                            st.dataframe(df, use_container_width=True)
                            st.markdown('**Raw sensitivity values:**')
                            st.write(df.to_dict(orient='list'))
                            table_shown = True
                        except Exception as e:
                            st.warning(f"Could not display table for {title}: {e}")
                if not table_shown:
                    st.info(f"No summary table found for {title}. Run the analysis to generate it.")
        else:
            st.info(f"You have selected: {hedge_type} hedge. Analysis and results will be shown here.")
    else:
        st.warning("No portfolio analysis results found. Please run a portfolio analysis first.")

st.markdown('<div class="footer-conference">Developed by Marcos Heredia Pimienta, Quantitative Risk Analyst', unsafe_allow_html=True)
