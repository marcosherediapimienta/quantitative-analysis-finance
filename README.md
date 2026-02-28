# option-pricing

Quant library for option pricing, Greeks, implied volatility, and simulation-based risk analysis.

## Current Structure

- `models`
  - Core pricing models and option contracts (CRR, Black-Scholes, Monte Carlo, Longstaff-Schwartz).
- `tools/analytics`
  - Greeks and implied volatility tooling.
- `data`
  - Market data adapters (historical volatility via Yahoo).
- `portfolio`
  - Portfolio pricing, aggregate Greeks, hedging, and sensitivity workflows.
- `risk`
  - Monte Carlo PnL simulation and risk metrics (VaR/ES).

## Refactor Conventions

- Keep modules focused on a single quant responsibility.
- Avoid hardcoded numerical values: use centralized configuration defaults.
- Avoid long `if/elif` chains: prefer enums, mappings, and composition.
- Use clear object-oriented APIs for pricing and analytics.

## Notes

Legacy folders were removed from the repository tree.
