# Risk Metrics for Portfolios of European Options

This README provides the theoretical background and logic behind the script `risk_metrics.py`, which calculates advanced risk metrics for portfolios of European options, including VaR, ES, stress testing, and Greeks analysis.

---

## 1. Introduction: Why Measure Risk in Options?

Options are nonlinear financial instruments, and their risk cannot be measured with traditional statistics alone. It is essential to quantify the portfolio's sensitivity to market movements (spot, volatility, rates) and estimate potential extreme losses.

---

## 2. Main Risk Metrics

### Value at Risk (VaR)
- **Definition:** VaR at confidence level α is the maximum expected loss over a given time horizon under normal market conditions.
- **Formula:**
  $$
  \mathrm{VaR}_{\alpha}(L) = \inf \{ l \in \mathbb{R} : P(L > l) \leq 1-\alpha \}
  $$
- **Interpretation:** A 1-day 99% VaR of €10,000 means there is a 1% probability of losing more than €10,000 in one day.

### Expected Shortfall (ES)
- **Definition:** ES (or CVaR) is the average loss in the worst cases, i.e., when the loss exceeds the VaR.
- **Formula:**
  $$
  \mathrm{ES}_{\alpha}(L) = \mathbb{E}[L\mid L > \mathrm{VaR}_{\alpha}(L)]
  $$
- **Advantage:** It is a coherent risk measure and better captures tail risk.

---

## 3. Simulation and Valuation of Option Portfolios

### Price Simulation
- Future scenarios for the underlying are simulated using Geometric Brownian Motion (GBM):
  $$
  S_T = S_0 \exp\left((r - 0.5\sigma^2)T + \sigma\sqrt{T}Z\right)
  $$
  where $Z \sim N(0,1)$.

### Option Valuation
- Several models are used: Black-Scholes, Binomial, Monte Carlo, and Finite Differences.
- The portfolio value is calculated by summing the value of each option under each scenario.

### P&L Calculation
- The P&L for each scenario is the difference between the simulated and current portfolio value.
- VaR and ES are calculated from the P&L distribution.

---

## 4. Greeks: Portfolio Sensitivities

- **Delta:** Sensitivity to spot.
- **Gamma:** Sensitivity of delta to spot.
- **Vega:** Sensitivity to volatility.
- **Theta:** Sensitivity to time decay.
- **Rho:** Sensitivity to interest rates.

The script calculates the aggregated Greeks for the portfolio to understand its global exposure.

---

## 5. Stress Testing

- The impact of extreme shocks in spot, volatility, and rates is evaluated.
- The P&L is calculated under each scenario and compared to the base value.
- Results are reported both with and without delta hedge (static directional hedge).

---

## 6. Delta Hedging

- Delta hedge simulates the coverage of the initial directional exposure of the portfolio.
- The delta-hedged P&L subtracts the impact of the underlying movement multiplied by the initial total delta.
- This isolates non-directional risk (gamma, vega, etc.).

---

## 7. Limitations and Extensions

- The script does **not** implement vega-hedge or rho-hedge, but it does calculate the necessary Greeks for such extensions.
- The framework is extensible to other Greeks and stress scenarios.

---

## 8. References
- Hull, J. (2018). Options, Futures, and Other Derivatives.
- Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk.
- McDonald, R. (2013). Derivatives Markets.

---

## 9. Recommended Usage
- Define your portfolio in the script.
- Run the analysis to obtain VaR, ES, Greeks, and stress testing.
- Analyze the results to understand the sensitivity and extreme risk of your options portfolio.

