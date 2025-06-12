# Quantitative Analysis Finance

A comprehensive Python toolkit for quantitative finance, focusing on option pricing, risk analysis, and portfolio management. This project provides robust implementations of key financial models, with both script-based and interactive (Streamlit) interfaces.

## Features

### Option Pricing

- **European and American Options**:  
  - **Black-Scholes Model**: Calculate prices and Greeks for European options.
  - **Binomial Model**: Suitable for both European and American options, providing flexibility in option pricing.
  - **Monte Carlo Simulation**: Offers a stochastic approach to option pricing, applicable to both European and American options.
  - **Finite Difference Method**: A numerical method for pricing options, particularly useful for complex derivatives.
- **Implied Volatility Calculation**: Determine the market's expectation of future volatility using the Black-Scholes model.
- **Greeks Calculation**: Compute Delta, Gamma, Vega, Theta, and Rho for all models to assess risk and sensitivity.
- **Model Comparison**: Visual and numerical comparison of pricing models to evaluate performance and accuracy.
- **Option Portfolios**: Construct and analyze portfolios of options to optimize risk and return.
- **Hedging Strategies**: Implement Delta, Delta+Gamma, and Vega hedging to manage portfolio risk.
- **Sensitivity Analysis**: Automated tools to analyze the impact of changes in key risk factors (Spot, interest rate, Volatility) on option pricing and portfolio value.

### Portfolio Management

- **Fundamental and Technical Analysis**: Evaluate securities using financial statements and market data.

### Data

- **Real-Time Data Integration**: Seamless integration with Yahoo Finance for up-to-date market data.

### Interactive App

- **Streamlit App**: A user-friendly interface for exploring and comparing financial models.
- **Comprehensive Analysis**: Supports full workflows for Black-Scholes, Binomial, and Monte Carlo models, including sensitivity analysis and hedging strategies.
- **Visualization and Reporting**: Generate and display plots and tables for sensitivity analysis and model comparison.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/quantitative-analysis-finance.git
   cd quantitative-analysis-finance
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option Pricing (Script)

- Run any script in the `option_pricing` submodules for interactive pricing and analysis.
- Example (European option, Binomial model):
  ```bash
  python option_pricing/binomial_model/european_options/european_binomial.py
  ```

## Directory Structure

```
quantitative-analysis-finance/
├── app.py
├── LICENSE
├── README.md
├── requirements.txt
├── option_pricing/
│   ├── black_scholes_model/
│   ├── binomial_model/
│   ├── monte_carlo/
│   ├── finite_difference_method/
│   ├── data/
│   └── visualizations/
├── portfolio_management/
│   ├── scripts/
│   ├── visualizations/
│   └── main.py
└── tests/
```

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.  
See the [LICENSE](LICENSE) file for details.

## Author

**Marcos Heredia Pimienta**  
MSc in Quantitative Finance

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Disclaimer

This software is for educational and research purposes. It is not intended for use in production or as financial advice.
