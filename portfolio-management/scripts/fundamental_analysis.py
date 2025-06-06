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
            return f"${value:,.0f}".replace(',', '.')  
        except (ValueError, TypeError):
            return "N/A"  
        
    def _print_values(self, metric, values):
        """Print values for a given metric."""
        if isinstance(values, pd.Series):
            print(f"  {metric}:")
            for date, value in values.items():
                formatted_date = date.strftime('%Y-%m-%d')
                formatted_value = self._format_currency(value)  
                print(f"    {formatted_date}: {formatted_value}")
        else:
            formatted_value = self._format_currency(values)  
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
        figures = []  # Store all figure objects

        for category, metrics in available_metrics.items():
            if metrics:
                fig, ax = plt.subplots(figsize=(10, 6))  
                for metric in metrics:
                    values = data.loc[metric]
                    if isinstance(values, pd.Series):
                        # Normalize the data before plotting
                        normalized_values = self._normalize_data(values)
                        sns.lineplot(x=values.index, y=normalized_values, label=metric, ax=ax)
                ax.set_title(f"{title} - {category}")
                ax.set_xlabel('Date')
                ax.set_ylabel('Normalized Value')
                ax.legend()
                ax.grid(True)
                figures.append(fig)  

                # Save the figure as a .png file in the visualizations directory
                plt.savefig(f'/home/marcos/Escritorio/mhp/quantitative-analysis-finance/portfolio-management/visualizations/{title}_{category}_plot.png')

        return figures  

    def get_balance_sheet(self, plot=False):
        """Retrieve and print the balance sheet for the given ticker."""
        try:
            info = self.company.balance_sheet
            if info.empty:
                raise ValueError("No balance sheet data available for the given ticker.")
            
            balance_sheet = {
                "Debt and Capital Structure": ['Total Debt', 'Net Debt', 'Long Term Debt', 'Current Debt'],
                "Capitalization and Equity": ['Total Capitalization', 'Total Equity Gross Minority Interest'],
                "Assets": ['Total Assets', 'Net PPE', 'Goodwill'],
                "Liabilities": ['Total Liabilities Net Minority Interest', 'Current Liabilities'],
                "Working Capital": ['Working Capital']
            }

            available_accounts = self._filter_metrics(info, balance_sheet)

            print(f"Balance Sheet for {self.ticker}:")
            for category, accounts in available_accounts.items():
                if accounts:
                    print(f"\n{category}:")
                    for account in accounts:
                        values = info.loc[account]
                        self._print_values(account, values)

            if plot:
                figures = self._plot_metrics(info, balance_sheet, "Balance Sheet")
                return figures  # Return figures for later display

        except Exception as e:
            print(f"Error retrieving balance sheet for {self.ticker}: {str(e)}")
            return []

    def get_income_statement(self, plot=False):
        """Retrieve and print the income statement for the given ticker."""
        try:
            info = self.company.financials
            if info.empty:
                raise ValueError("No income statement data available for the given ticker.")
            
            income_stmt = {
                "Revenue": ['Total Revenue', 'Cost Of Revenue'],
                "Gross and Operating Profitability": ['Gross Profit', 'Operating Income'],
                "Operating Profit and Pre-Tax": ['EBITDA', 'Earnings Before Interest and Taxes (EBIT)', 'Earnings Before Taxes (EBT)'],
                "Financial Profitability": ['Net Interest Income', 'Net Income']
            }

            available_metrics = self._filter_metrics(info, income_stmt)

            print(f"Income Statement for {self.ticker}:")
            for category, metrics in available_metrics.items():
                if metrics:
                    print(f"\n{category}:")
                    for metric in metrics:
                        values = info.loc[metric]
                        self._print_values(metric, values)

            if plot:
                figures = self._plot_metrics(info, income_stmt, "Income Statement")
                return figures  

        except Exception as e:
            print(f"Error retrieving income statement for {self.ticker}: {str(e)}")
            return []

    def get_cash_flow(self, plot=False):
        """Retrieve and print the cash flow statement for the given ticker."""
        try:
            info = self.company.cash_flow
            if info.empty:
                raise ValueError("No cash flow statement data available for the given ticker.")
            
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

            available_metrics = self._filter_metrics(info, cash_flow)

            print(f"Cash Flow Statement for {self.ticker}:")
            for category, metrics in available_metrics.items():
                if metrics:
                    print(f"\n{category}:")
                    for metric in metrics:
                        values = info.loc[metric]
                        self._print_values(metric, values)

            if plot:
                figures = self._plot_metrics(info, cash_flow, "Cash Flow Statement")
                return figures  

        except Exception as e:
            print(f"Error retrieving cash flow statement for {self.ticker}: {str(e)}")
            return []

    def get_financial_ratios(self):
        """Retrieve and print financial ratios for the given ticker."""
        try:
            info = self.company.info
            if not info:
                raise ValueError("No financial ratios data available for the given ticker.")
            
            ratios = {
                "Valuation Ratios": {
                    "P/E Ratio": info.get('trailingPE', 'N/A'),
                    "P/B Ratio": info.get('priceToBook', 'N/A'),
                    "P/S Ratio": info.get('priceToSalesTrailing12Months', 'N/A')
                },
                "Profitability Ratios": {
                    "ROE": info.get('returnOnEquity', 'N/A'),
                    "ROA": info.get('returnOnAssets', 'N/A'),
                    "Operating Margin": info.get('operatingMargins', 'N/A')
                },
                "Liquidity Ratios": {
                    "Current Ratio": info.get('currentRatio', 'N/A'),
                    "Quick Ratio": info.get('quickRatio', 'N/A')
                },
                "Debt Ratios": {
                    "Debt-to-Equity Ratio": info.get('debtToEquity', 'N/A'),
                }
            }

            print(f"Financial Ratios for {self.ticker}:")
            for category, ratio_dict in ratios.items():
                print(f"\n{category}:")
                for ratio, value in ratio_dict.items():
                    print(f"  {ratio}: {value}")

        except Exception as e:
            print(f"Error retrieving financial ratios for {self.ticker}: {str(e)}")

ticker = "NVDA"
analyzer = FinancialAnalyzer(ticker)
analyzer.get_balance_sheet(plot=True)
analyzer.get_income_statement(plot=True)
analyzer.get_cash_flow(plot=True)
analyzer.get_financial_ratios()