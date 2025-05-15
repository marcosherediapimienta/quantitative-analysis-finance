import streamlit as st

st.set_page_config(page_title="Volatilidad Implícita Black-Scholes", layout="centered")
st.title("Volatilidad Implícita (Black-Scholes)")

# Intentar importar los módulos de volatilidad implícita
bs_import_error = None
try:
    from black_scholes_model.european_options.call_implied_volatility import implied_volatility_call
    from black_scholes_model.european_options.put_implied_volatility import implied_volatility_put
except Exception as e:
    bs_import_error = str(e)

if bs_import_error:
    st.error(f"Error al importar funciones de Black-Scholes: {bs_import_error}")
else:
    st.write("Calcula la volatilidad implícita de una opción europea usando Black-Scholes.")
    option_type = st.selectbox("Tipo de opción", ["call", "put"])
    S = st.number_input("Spot price (S)", value=100.0)
    K = st.number_input("Strike price (K)", value=100.0)
    T = st.number_input("Time to maturity (años)", value=1.0)
    r = st.number_input("Risk-free rate (r, decimal)", value=0.05)
    market_price = st.number_input("Precio de mercado de la opción", value=10.0)
    if st.button("Calcular volatilidad implícita"):
        try:
            if option_type == "call":
                iv = implied_volatility_call(S, K, T, r, market_price)
            else:
                iv = implied_volatility_put(S, K, T, r, market_price)
            st.success(f"Volatilidad implícita ({option_type}): {iv:.4f}")
        except Exception as e:
            st.error(f"Error en el cálculo de volatilidad implícita: {e}")
