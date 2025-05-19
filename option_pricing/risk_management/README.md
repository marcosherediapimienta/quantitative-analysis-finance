# Risk Metrics: Value at Risk (VaR) y Expected Shortfall (ES) para Carteras de Opciones Europeas

Este script calcula el VaR y el ES de una cartera de opciones europeas (calls y puts) usando modelos de valoración y simulación de precios bajo dinámica de Black-Scholes.

# Teoría de VaR y ES: Explicación Paso a Paso

## 1. ¿Qué es el Value at Risk (VaR)?
El **VaR** es una medida de riesgo que estima la máxima pérdida potencial de una cartera durante un horizonte temporal específico y con un nivel de confianza dado.

- Ejemplo: Un VaR al 99% a 1 día de 100.000€ significa que hay un 1% de probabilidad de perder más de 100.000€ en un día.

**Fórmula:**
\[
VaR_{\alpha}(L) = \inf \{ l \in \mathbb{R} : P(L > l) \leq 1-\alpha \}
\]

## 2. ¿Qué es el Expected Shortfall (ES) o Conditional VaR?
El **ES** es la pérdida promedio en los peores casos, es decir, cuando la pérdida supera el VaR. Es una medida coherente de riesgo y captura mejor el riesgo de cola.

**Fórmula:**
\[
ES_{\alpha}(L) = E[L \mid L > VaR_{\alpha}(L)]
\]

## 3. ¿Cómo se calculan VaR y ES en la práctica?
- Se simula la distribución de pérdidas y ganancias (P&L) de la cartera bajo escenarios futuros.
- Se calcula el VaR como el percentil correspondiente de la distribución de P&L.
- El ES es la media de las pérdidas que superan el VaR.

## 4. ¿Por qué es importante el horizonte temporal?
- El VaR/ES siempre se refiere a un horizonte concreto (1 día, 10 días, etc.).
- En derivados, el valor de las opciones depende del tiempo a vencimiento, por lo que en los escenarios simulados se debe restar el horizonte al vencimiento real.

## 5. ¿Por qué usar simulación?
- Para carteras de derivados, la distribución de P&L no es normal ni simétrica.
- La simulación permite capturar la no linealidad y el efecto del paso del tiempo (theta decay).

## 6. Interpretación gráfica
- El VaR es el percentil de la cola izquierda del histograma de P&L.
- El ES es la media de las pérdidas más extremas (más allá del VaR).

## Lógica del Script Paso a Paso

1. **Definición de la Cartera**
   - Se define manualmente una lista de opciones, cada una con ticker, tipo (call/put), strike, tiempo a vencimiento (T, en años), y número de contratos.
   - Ejemplo:
     ```python
     portfolio = [
         {'ticker': '^SPX','type': 'call','K': 5955,'T': 0.0712,'contracts': 15},
         {'ticker': '^SPX','type': 'put','K': 5960,'T': 0.0712,'contracts': 10},
     ]
     ```

2. **Descarga de Precios y Cálculo de Volatilidad Histórica**
   - Se descargan precios históricos del subyacente con yfinance.
   - Se calcula la volatilidad histórica anualizada para cada subyacente.

3. **Cálculo de Volatilidad Implícita**
   - El usuario puede introducir el precio de mercado de cada opción.
   - Se calcula la volatilidad implícita usando el modelo de Black-Scholes.
   - Si no se introduce, se usa la volatilidad histórica.

5. **Simulación de Precios Futuros (GBM)**
   - Se simulan N escenarios de precios futuros del subyacente usando un movimiento browniano geométrico (GBM) para un horizonte definido (por ejemplo, 1 día: `HORIZONTE_VAR = 1/252`).

6. **Valoración de la Cartera en Escenarios Simulados**
   - Para cada escenario y modelo de pricing (Black-Scholes, Binomial, Monte Carlo, Diferencias Finitas), se valora la cartera:
     - El valor actual (V₀) se calcula con el vencimiento real de cada opción.
     - En cada escenario, el tiempo a vencimiento es `T_futuro = max(opt['T'] - HORIZONTE_VAR, 0)`.

7. **Cálculo de P&L, VaR y ES**
   - Se calcula el P&L como la diferencia entre el valor simulado y el valor actual de la cartera.
   - El VaR es el percentil de la cola izquierda de la distribución de P&L (por ejemplo, 1% para VaR al 99%).
   - El ES es la media de las pérdidas que superan el VaR.

8. **Visualización de Resultados**
   - Se grafican los histogramas de P&L simulados, marcando VaR y ES para cada modelo.
   - Se grafican también la distribución de precios simulados y de log-precios para validar la simulación.

## Visualizaciones generadas por el script

El script produce varias visualizaciones automáticas para facilitar el análisis del riesgo y la validación de la simulación:

### 1. Histograma de P&L simulado con VaR y ES
- Para cada modelo de valoración (Black-Scholes, Binomial, Monte Carlo, Diferencias Finitas), se grafica el histograma de las ganancias y pérdidas simuladas de la cartera.
- Se marcan el VaR (línea roja discontinua) y el ES (línea naranja discontinua) sobre el histograma.
- Archivo generado: `risk_metrics_results.png`

### 2. Histograma de P&L delta-hedgeado
- Se grafica el histograma de P&L de la cartera suponiendo cobertura delta (delta-hedge) estática.
- Permite comparar el riesgo residual tras neutralizar el riesgo direccional.
- Archivo generado: `risk_metrics_results_delta_hedge.png`

### 3. Distribución de precios simulados
- Para cada subyacente, se grafica la distribución de precios simulados al horizonte de VaR/ES.
- Permite validar la simulación bajo el modelo de movimiento browniano geométrico (GBM).
- Archivo generado: `simulated_prices_distribution.png`

### 4. Distribución de log-precios simulados
- Se grafica la distribución de los logaritmos de los precios simulados, que debe aproximar una normal bajo GBM.
- Archivo generado: `simulated_logprices_distribution.png`

#### Ejemplo de gráficos
- ![Histograma de P&L](risk_metrics_results.png)
- ![Histograma de P&L delta-hedgeado](risk_metrics_results_delta_hedge.png)
- ![Distribución de precios simulados](simulated_prices_distribution.png)
- ![Distribución de log-precios simulados](simulated_logprices_distribution.png)

Cada gráfico ayuda a interpretar visualmente el perfil de riesgo de la cartera y la validez de la simulación. Se recomienda revisar especialmente los histogramas de P&L y delta-hedge para entender el impacto de la cobertura y la exposición a riesgos residuales.

## Delta Hedging y VaR/ES Delta-Hedgeado

### Cálculo de Delta Total
El script ahora calcula el **delta total de la cartera** usando la fórmula de Black-Scholes para cada opción (ponderado por contratos). Esto permite conocer la sensibilidad de la cartera al movimiento del subyacente.

### Simulación de VaR/ES Delta-Hedgeado
- Se simula el P&L de la cartera suponiendo que se realiza un **delta hedging estático** al inicio del horizonte de simulación.
- En cada escenario, se resta la variación del subyacente multiplicada por el delta total inicial, simulando la cobertura con el activo subyacente.
- Se calcula y visualiza el **VaR** y **Expected Shortfall (ES)** de la cartera delta-hedgeada para cada modelo de pricing.
- El gráfico se guarda como `risk_metrics_results_delta_hedge.png`.

### Interpretación y advertencias
- **El delta hedging es relevante y efectivo principalmente en carteras con opciones ATM o ITM**, donde el delta es significativo. En carteras con opciones OTM, el delta es bajo y el hedge tiene poco impacto.
- **En condiciones normales**, el VaR/ES delta-hedgeado debe ser menor en magnitud (menos negativo) que el VaR/ES sin hedge, ya que se elimina el riesgo direccional al spot.
- **En escenarios de volatilidad extrema, gamma negativa o movimientos muy grandes del spot**, el VaR/ES delta-hedgeado puede ser más negativo que el original. El script advierte sobre este comportamiento y recomienda revisar la composición de la cartera y los parámetros de simulación.

### Ejemplo de output
```
Delta total de la cartera (Black-Scholes): 2.0936

--- VaR y ES para la cartera delta-hedgeada (neutralizada al spot inicial) ---
Modelo: black_scholes
  Value at Risk (VaR) delta-hedgeado al 99.0%: -3542.53
  Expected Shortfall (ES) delta-hedgeado al 99.0%: -3698.83
...
```

### Recomendaciones de uso
- Para analizar el efecto real del delta hedging, prueba con carteras que incluyan opciones ATM o ITM y volatilidades realistas.
- Si el VaR/ES delta-hedgeado resulta más negativo que el original, revisa la volatilidad, el horizonte de simulación y la composición de la cartera.

---

**Referencias:**
- Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk.
- Hull, J. (2018). Risk Management and Financial Institutions.
