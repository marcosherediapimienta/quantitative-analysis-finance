# Finite Difference Method for European Option Pricing

This document explains the theory behind the finite difference method (explicit scheme) for pricing European options, as implemented in this folder.

## 1. Black-Scholes PDE
The price of a European option under the Black-Scholes model satisfies the following partial differential equation (PDE):

$$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
$$

where:
- $V = V(S, t)$: option price as a function of underlying price $S$ and time $t$
- $\sigma$: volatility
- $r$: risk-free interest rate
- $T$: maturity

## 2. Discretization
We discretize the $S$ and $t$ axes:
- $S_i = i \cdot \Delta S$, $i = 0, 1, ..., M$
- $t_j = j \cdot \Delta t$, $j = 0, 1, ..., N$

The grid is built backwards in time, from maturity $T$ to $t=0$.

## 3. Explicit Finite Difference Scheme
The derivatives are approximated as:
- $\frac{\partial V}{\partial t} \approx \frac{V_i^{j+1} - V_i^j}{\Delta t}$
- $\frac{\partial V}{\partial S} \approx \frac{V_{i+1}^j - V_{i-1}^j}{2\Delta S}$
- $\frac{\partial^2 V}{\partial S^2} \approx \frac{V_{i+1}^j - 2V_i^j + V_{i-1}^j}{(\Delta S)^2}$

The update rule for each node is:

$$
V_i^j = V_i^{j+1} + \Delta t \left[ \frac{1}{2} \sigma^2 S_i^2 \frac{V_{i+1}^{j+1} - 2V_i^{j+1} + V_{i-1}^{j+1}}{(\Delta S)^2} + r S_i \frac{V_{i+1}^{j+1} - V_{i-1}^{j+1}}{2\Delta S} - r V_i^{j+1} \right]
$$

## 4. Boundary and Final Conditions
- **At maturity ($t = T$):**
    - Call: $V(S, T) = \max(S - K, 0)$
    - Put: $V(S, T) = \max(K - S, 0)$
- **At $S = 0$:**
    - Call: $V(0, t) = 0$
    - Put: $V(0, t) = K e^{-r(T-t)}$
- **At $S_{max}$:**
    - Call: $V(S_{max}, t) = S_{max} - K e^{-r(T-t)}$
    - Put: $V(S_{max}, t) = 0$

## 5. Algorithm Steps
1. Set up the grid in $S$ and $t$.
2. Apply the payoff at maturity.
3. Apply boundary conditions for all $t$.
4. Step backwards in time using the explicit update rule.
5. Interpolate to get the price for the actual $S_0$.

## 6. Stability
The explicit method is conditionally stable. For stability, the time step $\Delta t$ should be small enough:

$$
\Delta t < \frac{1}{\sigma^2 M^2}
$$

where $M$ is the number of price steps.

## 7. References
- Hull, J. C. "Options, Futures, and Other Derivatives"
- Wilmott, P. "Paul Wilmott Introduces Quantitative Finance"
- https://en.wikipedia.org/wiki/Finite_difference_method_for_option_pricing

---

This folder contains Python scripts implementing the above method for European call and put options, including implied volatility calculation and comparison with Black-Scholes analytical prices.
