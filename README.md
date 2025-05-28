# Quantitative Analysis Finance

A comprehensive Python toolkit for quantitative finance, focused on option pricing, risk analysis, and portfolio management. This project provides robust implementations of the most important models in modern finance, with both script-based and interactive (Streamlit) interfaces.

## Features

### Option Pricing

- **European Options**:  
  - Black-Scholes Model  
  - Binomial Model  
  - Monte Carlo Simulation  
  - Finite Difference Method  
- **American Options**:  
  - Binomial Model  
  - Monte Carlo (Longstaff-Schwartz)  
- **Implied Volatility Calculation** for all models
- **Greeks Calculation** (Delta, Gamma, Vega, Theta, Rho) for all models
- **Model Comparison**: Visual and numerical comparison of pricing models

### Portfolio Management

- **Mean-Variance Optimization** (Markowitz)
- **CAPM (Capital Asset Pricing Model)**
- **Fundamental and Technical Analysis**
- **Market Risk Analysis**
- **Portfolio Value-at-Risk (VaR) and Expected Shortfall (ES)**
- **Portfolio Greeks Calculation**

### Data

- Integration with Yahoo Finance for real-time data
- Automated retrieval of risk-free rates and option chains

### Interactive App

- **Streamlit App** for visual exploration and comparison of models
- User-friendly interface for both beginners and professionals
- **Automated Sensitivity Analysis**: For Black-Scholes, Binomial, and Monte Carlo models, the app allows you to generate and visualize sensitivity analysis for Spot, interest rate (r), and Volatility with a single click. Results are saved as both PNG plots and CSV summary tables, which are displayed directly in the app for all models.
- **Hedging & Sensitivity Section**: Analyze the impact of Delta, Delta+Gamma, and Vega hedging on your portfolio, and browse the corresponding risk and sensitivity outputs interactively.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quantitative-analysis-finance.git
   cd quantitative-analysis-finance
   ```
2. Install dependencies:
   ```bash
   pip install -r option_pricing/requirements.txt
   ```

## Usage

### Option Pricing (Script)

- Run any script in the `option_pricing` submodules for interactive pricing and analysis.
- Example (European option, Binomial model):
  ```bash
  python option_pricing/binomial_model/european_options/european_binomial.py
  ```

### Portfolio Management

- Example (Mean-Variance Optimization):
  ```bash
  python portfolio-management/scripts/Portfolio_Optimization.py
  ```

### Streamlit App

- Launch the interactive app:
  ```bash
  streamlit run app.py
  ```
- **Note:** The app supports full workflow for Black-Scholes, Binomial, and Monte Carlo models, including automated generation and display of sensitivity analysis (plots and tables) for all key risk factors.

---

## Directory Structure

```
option_pricing/
  ├── black_scholes_model/
  ├── binomial_model/
  ├── monte_carlo/
  ├── finite_difference_method/
  ├── app.py
  ├── requirements.txt
portfolio-management/
  ├── scripts/
  ├── main.py
LICENSE
README.md
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.  
See the [LICENSE](LICENSE) file for details.

---

## Author

**Marcos Heredia Pimienta**  
MSc in Quantitative Finance

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## Disclaimer

This software is for educational and research purposes. It is not intended for use in production or as financial advice.
