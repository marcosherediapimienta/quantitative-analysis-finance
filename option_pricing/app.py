import streamlit as st
import os
import time
from datetime import datetime
import importlib.util
import sys
import math

st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")
st.title("Option Pricing Dashboard · Marcos Heredia Pimienta — Quantitative Analyst")

st.markdown("""
Esta aplicación permite explorar y visualizar los modelos y resultados de la carpeta `option_pricing`.

- **Modelos Black-Scholes**
- **Opciones Call y Put**
- **Superficies de volatilidad**
- **Simulación Monte Carlo**
- **Comparativas**

*Desarrollado por Marcos Heredia Pimienta, Quantitative Analyst.*

Selecciona una sección del menú lateral para comenzar.
""")

menu = st.sidebar.selectbox(
    "Selecciona una sección:",
    [
        "Black-Scholes",
        "Opciones Call",
        "Opciones Put",
        "Monte Carlo",
        "Datos de Opciones",
        "Comparativas"
    ]
)

# Sección de ayuda mejorada
with st.sidebar.expander("ℹ️ Ayuda e instrucciones", expanded=False):
    st.markdown("""
    **¿Cómo usar la aplicación?**
    - Selecciona una sección en el menú lateral.
    - Haz clic en los botones para ejecutar los modelos y ver los resultados.
    - Si un modelo genera gráficos (por ejemplo, PNG), se mostrarán automáticamente.
    - Si ocurre un error, se mostrará el mensaje correspondiente.
    - Puedes recargar la página para limpiar la salida.
    
    **Requisitos:**
    - Python 3.8+
    - Instala dependencias con: `pip install streamlit numpy matplotlib pandas scipy`
    - Ejecuta la app con: `streamlit run app.py`
    """)
    st.markdown("""
    **¿Problemas comunes?**
    - Si no ves gráficos, asegúrate de que los scripts los generen en la carpeta correcta.
    - Si ves errores de dependencias, revisa que estén instaladas.
    """)
    st.markdown("""
    **Contacto:**
    - Para soporte, contacta a tu desarrollador o revisa el README del proyecto.
    """)

# Sección de información del proyecto
with st.sidebar.expander("ℹ️ Sobre el proyecto", expanded=False):
    st.markdown("""
    Proyecto de análisis cuantitativo de opciones financieras.
    Incluye modelos Black-Scholes, simulaciones Monte Carlo, superficies de volatilidad y más.
    """)
    st.markdown("Repositorio: [GitHub](https://github.com/) (actualiza con tu enlace real)")
    st.markdown("""
    ---
    **Autor:** Marcos Heredia Pimienta  
    **Rol:** Quantitative Analyst
    """)

def mostrar_output_y_imagenes(output, carpeta_img=None):
    if output.strip():
        st.code(output)
    # Mostrar imágenes PNG generadas recientemente en la carpeta indicada
    if carpeta_img and os.path.isdir(carpeta_img):
        imgs = [f for f in os.listdir(carpeta_img) if f.endswith('.png')]
        if imgs:
            st.subheader('Gráficos generados:')
            for img in sorted(imgs, key=lambda x: os.path.getmtime(os.path.join(carpeta_img, x)), reverse=True):
                img_path = os.path.join(carpeta_img, img)
                mod_time = datetime.fromtimestamp(os.path.getmtime(img_path)).strftime('%Y-%m-%d %H:%M:%S')
                st.image(img_path, caption=f"{img} (generado: {mod_time})", use_column_width=True)
        else:
            st.info('No se encontraron gráficos PNG en la carpeta.')

# Función de chequeo de resultados para mayor seguridad
def verificar_resultado(valor, nombre, min_esperado=None, max_esperado=None, tolerancia=None):
    if valor is None or (isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor))):
        st.error(f"El resultado de {nombre} es inválido (NaN o infinito). Revisa los parámetros.")
        return False
    if min_esperado is not None and valor < min_esperado:
        st.warning(f"El resultado de {nombre} ({valor:.4f}) es menor al mínimo esperado ({min_esperado}).")
    if max_esperado is not None and valor > max_esperado:
        st.warning(f"El resultado de {nombre} ({valor:.4f}) es mayor al máximo esperado ({max_esperado}).")
    if tolerancia is not None and abs(valor) < tolerancia:
        st.info(f"El resultado de {nombre} es muy pequeño (|valor| < {tolerancia}).")
    return True

# Utilidad para importar funciones de Black-Scholes desde los scripts existentes
spec_call = importlib.util.spec_from_file_location(
    "call_iv", os.path.join("black_scholes_model", "call_options", "call_implied_volatility.py")
)
call_iv = importlib.util.module_from_spec(spec_call)
sys.modules["call_iv"] = call_iv
spec_call.loader.exec_module(call_iv)

spec_put = importlib.util.spec_from_file_location(
    "put_iv", os.path.join("black_scholes_model", "put_options", "put_implied_volatility.py")
)
put_iv = importlib.util.module_from_spec(spec_put)
sys.modules["put_iv"] = put_iv
spec_put.loader.exec_module(put_iv)

# Black-Scholes interactivo
if menu == "Black-Scholes":
    st.header("Modelo Black-Scholes Interactivo")
    st.write("Calcula el precio y los griegos de una opción Call o Put cambiando los parámetros.")
    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Tipo de opción", ["call", "put"])
        S = st.number_input("Precio spot (S)", value=100.0, min_value=0.01)
        K = st.number_input("Strike (K)", value=100.0, min_value=0.01)
        T = st.number_input("Tiempo a vencimiento (años, T)", value=0.5, min_value=0.01, step=0.01)
    with col2:
        r = st.number_input("Tasa libre de riesgo (r)", value=0.03, min_value=0.0, step=0.001, format="%.4f")
        sigma = st.number_input("Volatilidad (σ)", value=0.2, min_value=0.0001, step=0.01, format="%.4f")
    if st.button("Calcular precio y griegos"):
        if option_type == "call":
            price = call_iv.black_scholes_call_price(S, K, T, r, sigma)
            greeks = call_iv.calculate_greeks(S, K, T, r, sigma)
        else:
            price = put_iv.black_scholes_put_price(S, K, T, r, sigma)
            greeks = put_iv.calculate_greeks(S, K, T, r, sigma)
        if verificar_resultado(price, "Precio opción", min_esperado=0):
            st.success(f"Precio {option_type.capitalize()}: {price:.4f}")
        st.write("**Griegos:**")
        for g, v in greeks.items():
            verificar_resultado(v, g)
        st.json(greeks)
        # Mostrar gráfico de griegos
        import matplotlib.pyplot as plt
        import io
        buf = io.BytesIO()
        if option_type == "call":
            call_iv.plot_greeks(S, K, T, r, sigma)
            plt.savefig(buf, format="png")
        else:
            put_iv.plot_greeks(S, K, T, r, sigma)
            plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        st.image(buf, caption="Gráficos de griegos", use_column_width=True)

# OPCIONES CALL INTERACTIVO
if menu == "Opciones Call":
    st.header("Opciones Call Interactivo")
    st.write("Calcula el precio, la volatilidad implícita y la superficie de volatilidad para opciones Call con parámetros personalizados.")
    ticker = st.text_input("Ticker (ej: AAPL)", value="AAPL")
    S = st.number_input("Precio spot (S)", value=100.0, min_value=0.01)
    K = st.number_input("Strike (K)", value=100.0, min_value=0.01)
    T = st.number_input("Tiempo a vencimiento (años, T)", value=0.5, min_value=0.01, step=0.01)
    r = st.number_input("Tasa libre de riesgo (r)", value=0.03, min_value=0.0, step=0.001, format="%.4f")
    C_market = st.number_input("Precio de mercado de la call", value=10.0, min_value=0.0)
    if st.button("Calcular Volatilidad Implícita Call"):
        try:
            iv = call_iv.implied_volatility_newton(C_market, S, K, T, r)
            if verificar_resultado(iv, "Volatilidad implícita", min_esperado=0, max_esperado=5):
                st.success(f"Volatilidad implícita: {iv:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
    if st.button("Superficie de Volatilidad Call (requiere internet)"):
        try:
            import importlib.util
            import io
            import matplotlib.pyplot as plt
            import numpy as np
            spec = importlib.util.spec_from_file_location("call_volsurf", os.path.join("black_scholes_model", "call_options", "call_volatility_surface.py"))
            call_volsurf = importlib.util.module_from_spec(spec)
            sys.modules["call_volsurf"] = call_volsurf
            spec.loader.exec_module(call_volsurf)
            df = call_volsurf.calculate_volatility_surface(ticker)
            st.dataframe(df)
            # Graficar superficie
            call_volsurf.plot_volatility_surface(df, save_path="call_vol_surface_tmp.png")
            st.image("call_vol_surface_tmp.png", caption="Superficie de volatilidad Call", use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# OPCIONES PUT INTERACTIVO
if menu == "Opciones Put":
    st.header("Opciones Put Interactivo")
    st.write("Calcula el precio, la volatilidad implícita y la superficie de volatilidad para opciones Put con parámetros personalizados.")
    ticker = st.text_input("Ticker (ej: AAPL)", value="AAPL", key="put_ticker")
    S = st.number_input("Precio spot (S)", value=100.0, min_value=0.01, key="put_S")
    K = st.number_input("Strike (K)", value=100.0, min_value=0.01, key="put_K")
    T = st.number_input("Tiempo a vencimiento (años, T)", value=0.5, min_value=0.01, step=0.01, key="put_T")
    r = st.number_input("Tasa libre de riesgo (r)", value=0.03, min_value=0.0, step=0.001, format="%.4f", key="put_r")
    P_market = st.number_input("Precio de mercado de la put", value=10.0, min_value=0.0, key="put_Pmkt")
    if st.button("Calcular Volatilidad Implícita Put"):
        try:
            iv = put_iv.implied_volatility_newton(P_market, S, K, T, r)
            if verificar_resultado(iv, "Volatilidad implícita", min_esperado=0, max_esperado=5):
                st.success(f"Volatilidad implícita: {iv:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
    if st.button("Superficie de Volatilidad Put (requiere internet)"):
        try:
            import importlib.util
            import io
            import matplotlib.pyplot as plt
            import numpy as np
            spec = importlib.util.spec_from_file_location("put_volsurf", os.path.join("black_scholes_model", "put_options", "put_volatility_surface.py"))
            put_volsurf = importlib.util.module_from_spec(spec)
            sys.modules["put_volsurf"] = put_volsurf
            spec.loader.exec_module(put_volsurf)
            df = put_volsurf.calculate_volatility_surface(ticker)
            st.dataframe(df)
            # Graficar superficie
            put_volsurf.plot_volatility_surface(df, save_path="put_vol_surface_tmp.png")
            st.image("put_vol_surface_tmp.png", caption="Superficie de volatilidad Put", use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# MONTE CARLO INTERACTIVO
if menu == "Monte Carlo":
    st.header("Simulación Monte Carlo Interactiva")
    st.write("Simula el precio de una opción europea (Black-Scholes) o Heston con parámetros personalizados.")
    modelo = st.selectbox("Modelo Monte Carlo", ["Black-Scholes", "Heston"])
    S0 = st.number_input("Precio inicial (S0)", value=130.0, min_value=0.01)
    K = st.number_input("Strike (K)", value=120.0, min_value=0.01)
    T = st.number_input("Tiempo a vencimiento (años, T)", value=38/365, min_value=0.01, step=0.01)
    r = st.number_input("Tasa libre de riesgo (r)", value=0.0421, min_value=0.0, step=0.001, format="%.4f")
    N = st.number_input("Nº simulaciones", value=50000, min_value=1000, step=1000)
    option_type = st.selectbox("Tipo de opción", ["call", "put"])
    if modelo == "Black-Scholes":
        sigma = st.number_input("Volatilidad (σ)", value=0.3, min_value=0.0001, step=0.01, format="%.4f")
        if st.button("Simular Black-Scholes"):
            import numpy as np
            import matplotlib.pyplot as plt
            import io
            Z = np.random.standard_normal(int(N))
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            if option_type == 'call':
                payoff = np.maximum(ST - K, 0)
            else:
                payoff = np.maximum(K - ST, 0)
            option_price = np.exp(-r * T) * np.mean(payoff)
            if verificar_resultado(option_price, "Precio opción Monte Carlo", min_esperado=0):
                st.success(f"Precio estimado de la opción {option_type}: {option_price:.4f}")
            # Gráfico de histogramas de precios finales
            fig, ax = plt.subplots()
            ax.hist(ST, bins=50, alpha=0.7)
            ax.set_title("Distribución de precios al vencimiento")
            ax.set_xlabel("Precio final")
            ax.set_ylabel("Frecuencia")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            st.image(buf, caption="Histograma de precios simulados", use_column_width=True)
    else:
        q = st.number_input("Dividendo (q)", value=0.04, min_value=0.0, step=0.001, format="%.4f")
        V0 = st.number_input("Varianza inicial (V0)", value=0.04, min_value=0.0001, step=0.01, format="%.4f")
        kappa = st.number_input("Kappa (reversión media)", value=2.0, min_value=0.01, step=0.01)
        theta = st.number_input("Theta (media largo plazo)", value=0.0625, min_value=0.0001, step=0.01, format="%.4f")
        xi = st.number_input("Volatilidad de la varianza (xi)", value=0.3, min_value=0.0001, step=0.01, format="%.4f")
        rho = st.number_input("Correlación (rho)", value=-0.6, min_value=-1.0, max_value=1.0, step=0.01)
        n_steps = st.number_input("Nº pasos (por año)", value=252, min_value=10, step=10)
        if st.button("Simular Heston"):
            import importlib.util
            import io
            import matplotlib.pyplot as plt
            import numpy as np
            spec = importlib.util.spec_from_file_location("heston", os.path.join("monte-carlo", "call_Heston.py"))
            heston = importlib.util.module_from_spec(spec)
            sys.modules["heston"] = heston
            spec.loader.exec_module(heston)
            S, V = heston.heston_monte_carlo(S0, K, T, r, q, V0, kappa, theta, xi, rho, int(N), int(n_steps))
            payoffs = np.maximum(S[-1] - K, 0) if option_type == 'call' else np.maximum(K - S[-1], 0)
            price_mc = np.exp(-r * T) * np.mean(payoffs)
            if verificar_resultado(price_mc, "Precio opción Heston", min_esperado=0):
                st.success(f"Precio Monte Carlo (Heston) {option_type}: {price_mc:.4f}")
            # Gráficos de trayectorias
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].plot(S[:, :10])
            axs[0].set_title("Trayectorias (Heston)")
            axs[0].set_xlabel("Tiempo")
            axs[0].set_ylabel("Precio")
            axs[1].plot(V[:, :10])
            axs[1].set_title("Trayectorias de Volatilidad")
            axs[1].set_xlabel("Tiempo")
            axs[1].set_ylabel("V")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            st.image(buf, caption="Trayectorias simuladas (Heston)", use_column_width=True)

if menu == "Datos de Opciones":
    st.header("Datos de Opciones")
    st.write("Visualiza los datos de opciones disponibles.")
    if st.button("Mostrar datos de opciones"):
        try:
            output = os.popen('python3 data/data_options.py').read()
            mostrar_output_y_imagenes(output, carpeta_img="data")
        except Exception as e:
            st.error(f"Error: {e}")

# COMPARATIVAS DE MODELOS
if menu == "Comparativas":
    st.header("Comparativa de Modelos de Valoración de Opciones")
    st.write("Compara el precio de una opción usando Black-Scholes, Monte Carlo y Heston con los mismos parámetros.")
    S = st.number_input("Precio spot (S)", value=100.0, min_value=0.01, key="cmp_S")
    K = st.number_input("Strike (K)", value=100.0, min_value=0.01, key="cmp_K")
    T = st.number_input("Tiempo a vencimiento (años, T)", value=0.5, min_value=0.01, step=0.01, key="cmp_T")
    r = st.number_input("Tasa libre de riesgo (r)", value=0.03, min_value=0.0, step=0.001, format="%.4f", key="cmp_r")
    sigma = st.number_input("Volatilidad (σ)", value=0.2, min_value=0.0001, step=0.01, format="%.4f", key="cmp_sigma")
    option_type = st.selectbox("Tipo de opción", ["call", "put"], key="cmp_type")
    N = st.number_input("Nº simulaciones Monte Carlo", value=50000, min_value=1000, step=1000, key="cmp_N")
    # Parámetros Heston
    q = st.number_input("Dividendo (q) Heston", value=0.04, min_value=0.0, step=0.001, format="%.4f", key="cmp_q")
    V0 = st.number_input("Varianza inicial (V0) Heston", value=0.04, min_value=0.0001, step=0.01, format="%.4f", key="cmp_V0")
    kappa = st.number_input("Kappa (reversión media) Heston", value=2.0, min_value=0.01, step=0.01, key="cmp_kappa")
    theta = st.number_input("Theta (media largo plazo) Heston", value=0.0625, min_value=0.0001, step=0.01, format="%.4f", key="cmp_theta")
    xi = st.number_input("Volatilidad de la varianza (xi) Heston", value=0.3, min_value=0.0001, step=0.01, format="%.4f", key="cmp_xi")
    rho = st.number_input("Correlación (rho) Heston", value=-0.6, min_value=-1.0, max_value=1.0, step=0.01, key="cmp_rho")
    n_steps = st.number_input("Nº pasos (por año) Heston", value=252, min_value=10, step=10, key="cmp_nsteps")
    if st.button("Comparar modelos"):
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        # Black-Scholes
        if option_type == "call":
            price_bs = call_iv.black_scholes_call_price(S, K, T, r, sigma)
        else:
            price_bs = put_iv.black_scholes_put_price(S, K, T, r, sigma)
        if verificar_resultado(price_bs, "Black-Scholes", min_esperado=0):
            pass
        # Monte Carlo
        Z = np.random.standard_normal(int(N))
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        if option_type == 'call':
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        price_mc = np.exp(-r * T) * np.mean(payoff)
        if verificar_resultado(price_mc, "Monte Carlo", min_esperado=0):
            pass
        # Heston
        spec = importlib.util.spec_from_file_location("heston", os.path.join("monte-carlo", "call_Heston.py"))
        heston = importlib.util.module_from_spec(spec)
        sys.modules["heston"] = heston
        spec.loader.exec_module(heston)
        S_h, V_h = heston.heston_monte_carlo(S, K, T, r, q, V0, kappa, theta, xi, rho, int(N), int(n_steps))
        if option_type == 'call':
            payoff_h = np.maximum(S_h[-1] - K, 0)
        else:
            payoff_h = np.maximum(K - S_h[-1], 0)
        price_heston = np.exp(-r * T) * np.mean(payoff_h)
        if verificar_resultado(price_heston, "Heston", min_esperado=0):
            pass
        # Mostrar resultados
        st.write("### Resultados de la comparativa")
        st.table({
            "Modelo": ["Black-Scholes", "Monte Carlo", "Heston"],
            "Precio": [f"{price_bs:.4f}", f"{price_mc:.4f}", f"{price_heston:.4f}"]
        })
        # Gráfico comparativo
        fig, ax = plt.subplots()
        modelos = ["Black-Scholes", "Monte Carlo", "Heston"]
        precios = [price_bs, price_mc, price_heston]
        ax.bar(modelos, precios, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_ylabel("Precio de la opción")
        ax.set_title("Comparativa de modelos")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        st.image(buf, caption="Comparativa de precios de modelos", use_column_width=True)

st.markdown("""
---
Desarrollado por **Marcos Heredia Pimienta, Quantitative Analyst**. Última actualización: Mayo 2025
""")
