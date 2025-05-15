# Binomial Model for European Option Pricing

## Overview
This module implements the Cox-Ross-Rubinstein (CRR) binomial tree model for pricing European call and put options. The binomial model is a discrete-time approach that models the evolution of the underlying asset price over a series of time steps, allowing for flexible and intuitive option pricing.

## Key Features
- **Interactive script**: Prompts the user for ticker, expiry, strike, and fetches spot and market price from Yahoo Finance.
- **Automatic implied volatility**: Calculates implied volatility (IV) from market price using Newton-Raphson and Brent's method. Uses IV by default in pricing.
- **User input fallback**: Only asks for volatility if IV cannot be calculated.
- **Professional output**: Displays all relevant parameters, binomial factors (u, d, p, dt), and compares model price to market price.
- **Robust error handling**: Handles missing data and invalid input gracefully.

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

## Usage
Run the script `binomial.py` interactively:

```bash
python binomial.py
```

You will be prompted for:
- Stock ticker (e.g., ^SPX)
- Expiration date (choose from available)
- Strike price (choose from available)
- Risk-free rate (default provided)
- Option type (call or put)
- Number of steps (N, default: days to expiry)

The script will:
- Fetch spot price and option chain from Yahoo Finance
- Retrieve market price for the selected option
- Calculate implied volatility (IV) if possible
- Use IV in the binomial model (or ask for volatility if IV is unavailable)
- Display all parameters, binomial factors, and compare model price to market price

## Output Example
```
==================================================
European Call Option Pricing (Binomial Model)
==================================================
Underlying:        ^SPX
Spot price (S):    5000.00
Strike (K):        5050.00
Expiration:        2025-06-21
Time to expiry:    37 days (0.1014 years)
Risk-free rate:    4.21%
Volatility (σ):    18.23%
Steps (N):         37
dt:                0.002740
u (up factor):     1.010563
d (down factor):   0.989553
p (risk-neutral):  0.507123
Market price:      75.3200
Model price:       75.3012
Difference:        -0.0188
==================================================
```

## Notes
- The model is for **European options** only (no early exercise).
- Increasing the number of steps (N) improves accuracy but increases computation time.
- The script is robust to missing data and will prompt for manual input if needed.

## References
- Cox, J.C., Ross, S.A., & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. Journal of Financial Economics, 7(3), 229-263.
- Hull, J. C. (Options, Futures, and Other Derivatives)
- [Wikipedia: Binomial options pricing model](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)

## File Structure
- `binomial.py`: Main interactive script for pricing European options using the binomial model.

---
For questions or improvements, contact the project maintainer.
