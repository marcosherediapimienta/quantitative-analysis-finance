import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FinancialAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.company = yf.Ticker(self.ticker)

    def _format_currency(self, value):
        """Format a numeric value as a currency string with dollar sign and dot as thousand separator."""
        try:
            return f"${value:,.0f}".replace(',', '.')  # Formato completo con separadores de miles
        except (ValueError, TypeError):
            return "N/A"  # Return "N/A" if the value is not numeric
        
    def _print_values(self, metric, values):
        """Print values for a given metric."""
        if isinstance(values, pd.Series):
            print(f"  {metric}:")
            for date, value in values.items():
                formatted_date = date.strftime('%Y-%m-%d')
                formatted_value = self._format_currency(value)  # Valor completo
                print(f"    {formatted_date}: {formatted_value}")
        else:
            formatted_value = self._format_currency(values)  # Valor completo
            print(f"  {metric}: {formatted_value}")

    def _filter_metrics(self, data, metrics_dict):
        """Filter metrics that exist in the financial data."""
        available_metrics = {key: [] for key in metrics_dict.keys()}
        for category, metrics in metrics_dict.items():
            for metric in metrics:
                if metric in data.index:
                    available_metrics[category].append(metric)
        return available_metrics

    def _normalize_data(self, data):
        """Normalize the data to a 0-1 scale."""
        return (data - data.min()) / (data.max() - data.min())

    def _plot_metrics(self, data, metrics_dict, title):
        """Plot the metrics that exist in the financial data."""
        available_metrics = self._filter_metrics(data, metrics_dict)
        for category, metrics in available_metrics.items():
            if metrics:
                plt.figure(figsize=(10, 6))
                for metric in metrics:
                    values = data.loc[metric]
                    if isinstance(values, pd.Series):
                        # Normalizar los datos antes de graficar
                        normalized_values = self._normalize_data(values)
                        sns.lineplot(x=values.index, y=normalized_values, label=metric)
                plt.title(f"{title} - {category}")
                plt.xlabel('Date')
                plt.ylabel('Normalized Value')
                plt.legend()
                plt.grid(True)
                # No llamamos a plt.show() aquí

    def get_balance_sheet(self, plot=False):
        try:
            # Get the balance sheet
            info = self.company.balance_sheet

            # List of accounts to filter
            balance_sheet = {
                "Debt and Capital Structure": [
                    'Total Debt',  # Total Debt
                    'Net Debt',  # Net Debt
                    'Long Term Debt',  # Long Term Debt
                    'Current Debt'  # Current Debt
                ],
                "Capitalization and Equity": [
                    'Total Capitalization',  # Total Capitalization
                    'Total Equity Gross Minority Interest'  # Total Equity with Minority Interest
                ],
                "Assets": [
                    'Total Assets',  # Total Assets
                    'Net PPE',  # Net Property, Plant, and Equipment
                    'Goodwill'  # Goodwill
                ],
                "Liabilities": [
                    'Total Liabilities Net Minority Interest',  # Total Liabilities with Minority Interest
                    'Current Liabilities'  # Current Liabilities
                ],
                "Working Capital": [
                    'Working Capital'  # Working Capital
                ]
            }

            # Filter accounts that exist in the balance sheet
            available_accounts = self._filter_metrics(info, balance_sheet)

            # Print the balance sheet with formatted values
            print(f"Balance Sheet for {self.ticker}:")
            for category, accounts in available_accounts.items():
                if accounts:
                    print(f"\n{category}:")
                    for account in accounts:
                        values = info.loc[account]
                        self._print_values(account, values)

            # Plot the balance sheet metrics if requested
            if plot:
                self._plot_metrics(info, balance_sheet, "Balance Sheet")

        except Exception as e:
            print(f"Error retrieving balance sheet for {self.ticker}: {e}")

    def get_income_statement(self, plot=False):
        try:
            # Get the income statement
            info = self.company.financials

            # List of metrics to filter
            income_stmt = {
                "Revenue": [
                    'Total Revenue',  # Total Revenue
                    'Cost Of Revenue'  # Cost of Revenue
                ],
                "Gross and Operating Profitability": [
                    'Gross Profit',  # Gross Profit
                    'Operating Income'  # Operating Income (EBIT)
                ],
                "Operating Profit and Pre-Tax": [
                    'EBITDA',  # EBITDA
                    'Earnings Before Interest and Taxes (EBIT)',  # EBIT
                    'Earnings Before Taxes (EBT)'  # Earnings Before Taxes
                ],
                "Financial Profitability": [
                    'Net Interest Income',  # Net Interest Income
                    'Net Income'  # Net Income
                ]
            }

            # Filter metrics that exist in the income statement
            available_metrics = self._filter_metrics(info, income_stmt)

            # Print the income statement with formatted values
            print(f"Income Statement for {self.ticker}:")
            for category, metrics in available_metrics.items():
                if metrics:
                    print(f"\n{category}:")
                    for metric in metrics:
                        values = info.loc[metric]
                        self._print_values(metric, values)

            # Plot the income statement metrics if requested
            if plot:
                self._plot_metrics(info, income_stmt, "Income Statement")

        except Exception as e:
            print(f"Error retrieving income statement for {self.ticker}: {e}")

    def get_cash_flow(self, plot=False):
        try:
            # Get the cash flow statement
            info = self.company.cash_flow

            # List of metrics to filter
            cash_flow = {
                "Cash Flow": [
                    'Operating Cash Flow',  
                    'Investing Cash Flow',
                    'Financing Cash Flow',
                    'Capital Expenditure',
                    'Free Cash Flow',
                    'Repayment Debt',
                    'Repurchase of Capital Stock'
                ]
            }

            # Filter metrics that exist in the cash flow statement
            available_metrics = self._filter_metrics(info, cash_flow)

            # Print the cash flow statement with formatted values
            print(f"Cash Flow Statement for {self.ticker}:")
            for category, metrics in available_metrics.items():
                if metrics:
                    print(f"\n{category}:")
                    for metric in metrics:
                        values = info.loc[metric]
                        self._print_values(metric, values)

            # Plot the cash flow metrics if requested
            if plot:
                self._plot_metrics(info, cash_flow, "Cash Flow Statement")

        except Exception as e:
            print(f"Error retrieving cash flow statement for {self.ticker}: {e}")