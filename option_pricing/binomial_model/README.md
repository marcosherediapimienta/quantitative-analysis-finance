# Binomial Model for European Option Pricing

## Overview
This module implements the Cox-Ross-Rubinstein (CRR) binomial tree model for pricing European call and put options. The binomial model is a discrete-time approach that models the evolution of the underlying asset price over a series of time steps, allowing for flexible and intuitive option pricing.


## Theoretical Background
The binomial model assumes that at each time step, the underlying asset price can move up by a factor `u` or down by a factor `d`. The risk-neutral probability `p` is used to discount expected payoffs back to present value. As the number of steps increases, the binomial model converges to the Black-Scholes price for European options.

- **Up factor (u):** `u = exp(sigma * sqrt(dt))`
- **Down factor (d):** `d = 1 / u`
- **Risk-neutral probability (p):** `p = (exp(r * dt) - d) / (u - d)`
- **Time step (dt):** `dt = T / N`

Where:
- `sigma`: volatility
- `r`: risk-free rate
- `T`: time to maturity (years)
- `N`: number of steps

## Model Assumptions
The Cox-Ross-Rubinstein binomial model relies on several key assumptions:

- The underlying asset price follows a multiplicative binomial process (can move up by factor `u` or down by factor `d` at each step).
- No dividends are paid during the life of the option (unless explicitly modeled).
- Markets are frictionless: no transaction costs or taxes, and assets are perfectly divisible.
- The risk-free interest rate (`r`) is constant and known for the duration of the option.
- Volatility (`sigma`) is constant and known (or implied from market prices).
- Trading of the underlying asset and option is continuous, and short selling is allowed.
- There are no arbitrage opportunities.
- The option is European style: it can only be exercised at expiration.

## Example: Binomial Tree Diagram
Below is a sample diagram of a binomial tree for the underlying asset price evolution (for N=3 steps):

```
        S
       / \
    uS   dS
    / \   / \
 u^2S d u dS d^2S
   ...   ...
```

For larger N, the tree grows in width and depth. The script also generates and saves a graphical plot of the binomial tree as `binomial_tree.png` each time you run it, using your selected parameters.

![Binomial Tree Example](binomial_tree.png)

## Notes
- The model is for **European options** only (no early exercise).
- Increasing the number of steps (N) improves accuracy but increases computation time.
- The script is robust to missing data and will prompt for manual input if needed.

## References
- Cox, J.C., Ross, S.A., & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. Journal of Financial Economics, 7(3), 229-263.
- Hull, J. C. (Options, Futures, and Other Derivatives)
- [Wikipedia: Binomial options pricing model](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)

