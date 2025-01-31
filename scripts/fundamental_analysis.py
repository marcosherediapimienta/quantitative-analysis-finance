import yfinance as yf
import pandas as pd

class BalanceSheetAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker

    def _format_currency(self, value):
        """Format a numeric value as a currency string with dollar sign and dot as thousand separator."""
        try:
            return f"${value:,.0f}".replace(',', '.')
        except (ValueError, TypeError):
            return "N/A"  # Return "N/A" if the value is not numeric

    def get_balance_sheet(self):
        try:
            # Download company data from Yahoo Finance
            company = yf.Ticker(self.ticker)

            # Get the balance sheet
            info = company.balance_sheet

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
            available_accounts = {key: [] for key in balance_sheet.keys()}
            for category, accounts in balance_sheet.items():
                for account in accounts:
                    if account in info.index:
                        available_accounts[category].append(account)

            # Print the balance sheet with formatted values
            print(f"Balance Sheet for {self.ticker}:")
            for category, accounts in available_accounts.items():
                if accounts:
                    print(f"\n{category}:")
                    for account in accounts:
                        values = info.loc[account]

                        # If the value is a Series (multiple dates), iterate through each date
                        if isinstance(values, pd.Series):
                            print(f"  {account}:")
                            for date, value in values.items():
                                formatted_value = self._format_currency(value)
                                print(f"    {date}: {formatted_value}")
                        else:
                            formatted_value = self._format_currency(values)
                            print(f"  {account}: {formatted_value}")
        
        except Exception as e:
            print(f"Error retrieving balance sheet for {self.ticker}: {e}")

    def get_income_statement(self):
        try:
            # Download company data from Yahoo Finance
            company = yf.Ticker(self.ticker)

            # Get the income statement
            info = company.financials  # Note that 'income_stmt' might not be the correct property

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
            available_metrics = {key: [] for key in income_stmt.keys()}
            for category, metrics in income_stmt.items():
                for metric in metrics:
                    if metric in info.index:
                        available_metrics[category].append(metric)

            # Print the income statement with formatted values
            print(f"Income Statement for {self.ticker}:")
            for category, metrics in available_metrics.items():
                if metrics:
                    print(f"\n{category}:")
                    for metric in metrics:
                        values = info.loc[metric]

                        # If the value is a Series with multiple dates, iterate and print each value
                        if isinstance(values, pd.Series):
                            print(f"  {metric}:")
                            for date, value in values.items():
                                # Ensure monetary format
                                formatted_value = f"${value:,.0f}".replace(',', '.')
                                print(f"    {date}: {formatted_value}")
                        else:
                            formatted_value = f"${values:,.0f}".replace(',', '.')
                            print(f"  {metric}: {formatted_value}")

        except Exception as e:
            print(f"Error retrieving income statement for {self.ticker}: {e}")
