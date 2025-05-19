# Value at Risk (VaR) y Expected Shortfall (ES): Teoría Básica

## Value at Risk (VaR)
El **Value at Risk (VaR)** es una medida de riesgo utilizada para estimar la máxima pérdida potencial de una cartera financiera durante un horizonte temporal específico y con un nivel de confianza dado. Formalmente, el VaR al nivel de confianza \( \alpha \) (por ejemplo, 99%) responde a la pregunta:

> "¿Cuál es la pérdida máxima que no se superará con una probabilidad del 99% en el periodo considerado?"

Matemáticamente, para una variable aleatoria de pérdidas \( L \):

\[
VaR_{\alpha}(L) = \inf \{ l \in \mathbb{R} : P(L > l) \leq 1-\alpha \}
\]

- Si el VaR a 1 día y 99% es 100.000€, significa que hay un 1% de probabilidad de perder más de 100.000€ en un día.
- El VaR **no** indica la magnitud de las pérdidas más allá de ese umbral.

## Expected Shortfall (ES) / Conditional VaR (CVaR)
El **Expected Shortfall (ES)**, también conocido como **Conditional VaR (CVaR)**, es una medida de riesgo que estima la pérdida promedio en los peores casos, es decir, cuando la pérdida supera el VaR. Es una medida coherente de riesgo y captura mejor el riesgo de cola que el VaR.

Matemáticamente:

\[
ES_{\alpha}(L) = E[L \mid L > VaR_{\alpha}(L)]
\]

- El ES al 99% es el valor esperado de las pérdidas que exceden el VaR al 99%.
- El ES siempre es mayor o igual que el VaR para el mismo nivel de confianza.

## Interpretación y Uso
- **VaR** es útil para comunicar el riesgo máximo esperado bajo condiciones normales de mercado.
- **ES** es preferido por reguladores y gestores de riesgo porque considera la magnitud de las pérdidas extremas.
- Ambas métricas requieren la simulación o modelado de la distribución de pérdidas y ganancias (P&L) de la cartera.

## Ejemplo Gráfico
En una distribución de P&L simulada:
- El VaR es el percentil (por ejemplo, 1%) de la cola izquierda.
- El ES es el promedio de las pérdidas que están más allá de ese percentil.

---

**Referencias:**
- Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk.
- Hull, J. (2018). Risk Management and Financial Institutions.
