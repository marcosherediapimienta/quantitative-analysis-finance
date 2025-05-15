# Monte Carlo Simulation for Option Pricing

## Overview
Monte Carlo simulation is a numerical method used to estimate the price of financial derivatives, such as European options, by simulating the random evolution of the underlying asset price under risk-neutral dynamics. It is especially useful when analytical solutions (like Black-Scholes) are unavailable or for more complex payoffs.

## Theoretical Background

### 1. Risk-Neutral Valuation
Under the risk-neutral measure, the price of a European option is the discounted expected value of its payoff at maturity:

    Price = E_Q[exp(-rT) * Payoff(ST)]

where:
- `E_Q` denotes expectation under the risk-neutral measure
- `r` is the risk-free rate
- `T` is the time to maturity
- `ST` is the underlying asset price at maturity
- `Payoff(ST)` is the option payoff (e.g., max(ST - K, 0) for a call)

### 2. Asset Price Simulation (Black-Scholes Model)
The underlying asset price is assumed to follow a geometric Brownian motion:

    dS = r S dt + sigma S dW

where:
- `S` is the asset price
- `r` is the risk-free rate
- `sigma` is the volatility
- `dW` is a Wiener process (Brownian motion)

The solution for the asset price at maturity is:

    ST = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

where:
- `S0` is the initial asset price
- `Z` is a standard normal random variable (Z ~ N(0,1))

### 3. Monte Carlo Algorithm Steps
1. **Simulate N random paths** for the underlying asset price at maturity using the formula above.
2. **Compute the payoff** for each simulated path.
3. **Average the payoffs** and discount them to present value:

       Price ≈ exp(-rT) * (1/N) * sum_{i=1}^N Payoff_i

4. **Estimate error**: The standard error decreases as 1/sqrt(N).

### 4. Advantages and Limitations
- **Advantages:**
  - Flexible for complex payoffs and path-dependent options.
  - Easy to implement for high-dimensional problems.
- **Limitations:**
  - Slow convergence: requires many simulations for high accuracy.
  - Not efficient for American options (early exercise).

## Typical Payoff Functions
- **Call option:** max(ST - K, 0)
- **Put option:** max(K - ST, 0)

## Practical Notes
- The accuracy improves with more simulations (N), but computation time increases.
- Using the implied volatility from market prices as input makes the model price comparable to observed market prices.
- Random number generation should be seeded for reproducibility.

## References
- Hull, J. C. (Options, Futures, and Other Derivatives)
- Glasserman, P. (Monte Carlo Methods in Financial Engineering)
- [Wikipedia: Monte Carlo methods for option pricing](https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing)
