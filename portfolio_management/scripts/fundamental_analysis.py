import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class FinancialAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visualizations")
        os.makedirs(self.VIS_DIR, exist_ok=True)
        self.company = yf.Ticker(self.ticker)

    def _format_currency(self, value):
        """Format a numeric value as a currency string with dollar sign and dot as thousand separator."""
        try:
            if pd.isna(value):
                return "N/A"
            return f"${value:,.0f}".replace(',', '.')  
        except (ValueError, TypeError):
            return "N/A"
        
    def _print_values(self, metric, values):
        """Print values for a given metric, excluding the year 2020."""
        if isinstance(values, pd.Series):
            print(f"  {metric}:")
            for date, value in values.items():
                if date.year != 2020:  # Exclude the year 2020
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
        """Plot each metric separately that exists in the financial data."""
        available_metrics = self._filter_metrics(data, metrics_dict)
        figures = []  # Store all figure objects

        for category, metrics in available_metrics.items():
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(10, 6))  
                values = data.loc[metric]
                if isinstance(values, pd.Series):
                    # Normalize the data before plotting
                    normalized_values = self._normalize_data(values)
                    sns.lineplot(x=values.index, y=normalized_values, label=metric, ax=ax)
                ax.set_title(f"{title} - {metric}")
                ax.set_xlabel('Date')
                ax.set_ylabel('Normalized Value')
                ax.legend()
                ax.grid(True)
                figures.append(fig)  

                # Save the figure as a .png file
                plt.savefig(os.path.join(self.VIS_DIR, f'{title}_{metric}_plot.png'))
                plt.close(fig)  # Close the figure after saving

        return figures  

    def _print_header(self, title):
        """Print a formatted header for sections."""
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")

    def _print_metric(self, name, value, unit=""):
        """Print a formatted metric with alignment."""
        if isinstance(value, float):
            print(f"{name:<30} {value:.2f}{unit}")
        else:
            print(f"{name:<30} {value}")

    def get_balance_sheet(self, plot=False):
        """Retrieve and print the balance sheet for the given ticker."""
        try:
            info = self.company.balance_sheet
            if info.empty:
                raise ValueError("No balance sheet data available for the given ticker.")
            
            balance_sheet = {
                "Debt and Capital Structure": ['Total Debt', 'Net PPE'],
                "Capitalization and Equity": ['Total Equity Gross Minority Interest', 'Retained Earnings', 'Common Stock Equity'],
                "Assets": ['Total Assets', 'Cash And Cash Equivalents'],
                "Liabilities": ['Total Liabilities Net Minority Interest'],
                "Working Capital": ['Working Capital']
            }

            available_accounts = self._filter_metrics(info, balance_sheet)

            self._print_header(f"Balance Sheet for {self.ticker}")
            for category, accounts in available_accounts.items():
                if accounts:
                    print(f"\n{category}:")
                    for account in accounts:
                        values = info.loc[account]
                        self._print_values(account, values)

            if plot:
                figures = self._plot_metrics(info, balance_sheet, "Balance Sheet")
                if not info.empty:
                    selected_info = info.loc[sum(balance_sheet.values(), [])]
                    selected_info.to_csv(os.path.join(self.VIS_DIR, 'balance_sheet.csv'))
                return figures

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
                "Revenue": ['Total Revenue'],
                "Gross and Operating Profitability": ['Gross Profit', 'Operating Income'],
                "Operating Profit and Pre-Tax": ['EBITDA', 'EBIT'],
                "Financial Profitability": ['Net Income'],
                "Tax and EPS": ['Tax Provision', 'Diluted EPS']
            }

            available_metrics = self._filter_metrics(info, income_stmt)

            self._print_header(f"Income Statement for {self.ticker}")
            for category, metrics in available_metrics.items():
                if metrics:
                    print(f"\n{category}:")
                    for metric in metrics:
                        values = info.loc[metric]
                        self._print_values(metric, values)

            if plot:
                figures = self._plot_metrics(info, income_stmt, "Income Statement")
                if not info.empty:
                    selected_info = info.loc[sum(income_stmt.values(), [])]
                    selected_info.to_csv(os.path.join(self.VIS_DIR, 'income_statement.csv'))
                return figures  

        except Exception as e:
            print(f"Error retrieving income statement for {self.ticker}: {str(e)}")
            return []

    def get_cash_flow(self, plot=False):
        """Retrieve and print the cash flow statement for the given ticker, with combined plots for specific metrics."""
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
                    'End Cash Position',
                    'Changes In Cash'
                ]
            }

            available_metrics = self._filter_metrics(info, cash_flow)

            self._print_header(f"Cash Flow Statement for {self.ticker}")
            for category, metrics in available_metrics.items():
                if metrics:
                    print(f"\n{category}:")
                    for metric in metrics:
                        values = info.loc[metric]
                        self._print_values(metric, values)

            if plot:
                figures = self._plot_metrics(info, cash_flow, "Cash Flow Statement")
                
                # Create combined plots for specific metrics
                combined_figures = []
                # Operating Cash Flow vs. Free Cash Flow
                if 'Operating Cash Flow' in info.index and 'Free Cash Flow' in info.index:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(x=info.columns, y=self._normalize_data(info.loc['Operating Cash Flow']), label='Operating Cash Flow', ax=ax)
                    sns.lineplot(x=info.columns, y=self._normalize_data(info.loc['Free Cash Flow']), label='Free Cash Flow', ax=ax)
                    ax.set_title("Operating vs Free Cash Flow")
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Normalized Value')
                    ax.legend()
                    ax.grid(True)
                    combined_figures.append(fig)
                    plt.savefig(os.path.join(self.VIS_DIR, 'Operating_vs_Free_Cash_Flow_plot.png'))
                    plt.close(fig)  # Close the figure after saving

                # Investing Cash Flow vs. Financing Cash Flow
                if 'Investing Cash Flow' in info.index and 'Financing Cash Flow' in info.index:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(x=info.columns, y=self._normalize_data(info.loc['Investing Cash Flow']), label='Investing Cash Flow', ax=ax)
                    sns.lineplot(x=info.columns, y=self._normalize_data(info.loc['Financing Cash Flow']), label='Financing Cash Flow', ax=ax)
                    ax.set_title("Investing vs Financing Cash Flow")
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Normalized Value')
                    ax.legend()
                    ax.grid(True)
                    combined_figures.append(fig)
                    plt.savefig(os.path.join(self.VIS_DIR, 'Investing_vs_Financing_Cash_Flow_plot.png'))
                    plt.close(fig)  # Close the figure after saving

                # End Cash Position vs. Changes In Cash
                if 'End Cash Position' in info.index and 'Changes In Cash' in info.index:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(x=info.columns, y=self._normalize_data(info.loc['End Cash Position']), label='End Cash Position', ax=ax)
                    sns.lineplot(x=info.columns, y=self._normalize_data(info.loc['Changes In Cash']), label='Changes In Cash', ax=ax)
                    ax.set_title("End Cash Position vs Changes In Cash")
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Normalized Value')
                    ax.legend()
                    ax.grid(True)
                    combined_figures.append(fig)
                    plt.savefig(os.path.join(self.VIS_DIR, 'End_Cash_Position_vs_Changes_In_Cash_plot.png'))
                    plt.close(fig)  # Close the figure after saving

                if not info.empty:
                    selected_info = info.loc[sum(cash_flow.values(), [])]
                    selected_info.to_csv(os.path.join(self.VIS_DIR, 'cash_flow.csv'))
                return figures + combined_figures

        except Exception as e:
            print(f"Error retrieving cash flow statement for {self.ticker}: {str(e)}")
            return []

    def get_financial_ratios(self):
        """Retrieve and print financial ratios for the given ticker, handling NaNs."""
        try:
            info = self.company.info
            if not info:
                raise ValueError("No financial ratios data available for the given ticker.")
            
            ratios = {
                "Valuation Ratios": {
                    "P/E Ratio": info.get('trailingPE'),
                    "P/B Ratio": info.get('priceToBook'),
                    "P/S Ratio": info.get('priceToSalesTrailing12Months'),
                    "Forward P/E": info.get('forwardPE')
                },
                "Profitability Ratios": {
                    "ROE": info.get('returnOnEquity'),
                    "ROA": info.get('returnOnAssets'),
                    "Operating Margin": info.get('operatingMargins'),
                    "Gross Margin": info.get('grossMargins')
                },
                "Liquidity Ratios": {
                    "Current Ratio": info.get('currentRatio'),
                    "Quick Ratio": info.get('quickRatio')
                },
                "Debt Ratios": {
                    "Debt-to-Equity Ratio": info.get('debtToEquity') / 100 if info.get('debtToEquity') is not None else None
                }
            }

            return ratios

        except Exception as e:
            return f"Error retrieving financial ratios for {self.ticker}: {str(e)}"

    def get_dividend_analysis(self):
        """Analyze dividend payments and yield"""
        try:
            div_info = self.company.dividends
            info = self.company.info
            
            analysis = {}
            current_yield = info.get('dividendYield', 'N/A')
            analysis['Current Dividend Yield'] = f"{current_yield if isinstance(current_yield, float) else 'N/A'}%"
            
            payout_ratio = info.get('payoutRatio', 'N/A')
            analysis['Payout Ratio'] = f"{payout_ratio*100 if isinstance(payout_ratio, float) else 'N/A'}%"
            
            if not div_info.empty:
                div_history = div_info.tail(5)
                analysis['Dividend History'] = {date.strftime('%Y-%m-%d'): f"${amount:.4f}" for date, amount in div_history.items()}
            else:
                analysis['Dividend History'] = "No dividend history available"

            return analysis

        except Exception as e:
            return f"Error retrieving dividend info: {str(e)}"

    def get_growth_metrics(self):
        """Analyze growth metrics"""
        try:
            info = self.company.info
            
            growth_metrics = {
                "Revenue Growth (YoY)": info.get('revenueGrowth', 'N/A'),
                "Earnings Growth (YoY)": info.get('earningsGrowth', 'N/A'),
                "Next 5 Years Growth Estimate": info.get('earningsQuarterlyGrowth', 'N/A')
            }
            
            return growth_metrics

        except Exception as e:
            return f"Error retrieving growth metrics: {str(e)}"

    def get_risk_metrics(self):
        """Analyze risk-related metrics"""
        try:
            info = self.company.info
            
            risk_metrics = {
                "Beta (Volatility)": info.get('beta', 'N/A'),
                "52-Week High": self._format_currency(info.get('fiftyTwoWeekHigh', 'N/A')),
                "52-Week Low": self._format_currency(info.get('fiftyTwoWeekLow', 'N/A')),
                "Short Ratio": info.get('shortRatio', 'N/A'),
                "Short % of Float": info.get('shortPercentOfFloat', 'N/A'),
                "Average Volume": self._format_currency(info.get('averageVolume', 'N/A'))
            }
            
            return risk_metrics

        except Exception as e:
            return f"Error retrieving risk metrics: {str(e)}"

    def get_additional_metrics(self):
        """Get additional important metrics"""
        try:
            info = self.company.info
            
            additional_metrics = {
                "Market Cap": self._format_currency(info.get('marketCap')),
                "Enterprise Value": self._format_currency(info.get('enterpriseValue')),
                "Shares Outstanding": self._format_currency(info.get('sharesOutstanding')),
                "Float": self._format_currency(info.get('floatShares')),
                "Institutional Ownership": f"{info.get('heldPercentInstitutions', 'N/A')*100:.2f}%" if isinstance(info.get('heldPercentInstitutions'), float) else 'N/A',
                "Insider Ownership": f"{info.get('heldPercentInsiders', 'N/A')*100:.2f}%" if isinstance(info.get('heldPercentInsiders'), float) else 'N/A',
                "Forward P/E": info.get('forwardPE'),
            }
            
            return additional_metrics

        except Exception as e:
            return f"Error retrieving additional metrics for {self.ticker}: {str(e)}"

    def calculate_cagr(self, series, years):
        """Calculate Compound Annual Growth Rate, ensuring sufficient data."""
        if len(series.dropna()) < 2:  # Ensure there are at least two non-NaN data points
            return "N/A (insufficient data)"
        try:
            start = series.dropna().iloc[-1]
            end = series.dropna().iloc[0]
            return (end/start)**(1/years) - 1
        except:
            return "N/A"

    def get_trend_analysis(self):
        """Analyze growth trends for key metrics"""
        try:
            # Get financial data
            bs = self.company.balance_sheet
            is_ = self.company.financials
            cf = self.company.cash_flow
            
            self._print_header(f"Trend Analysis for {self.ticker}")
            
            # Calculate number of years available
            num_years = len(bs.columns) if not bs.empty else 0
            
            if num_years >= 2:
                # Revenue growth
                if 'Total Revenue' in is_.index:
                    rev_growth = self.calculate_cagr(is_.loc['Total Revenue'], num_years)
                    self._print_metric(f"Revenue CAGR ({num_years} years):", rev_growth*100, "%")
                
                # EPS growth
                if 'Net Income' in is_.index and 'sharesOutstanding' in self.company.info:
                    net_income = is_.loc['Net Income']
                    shares = self.company.info['sharesOutstanding']
                    eps = net_income / shares
                    eps_growth = self.calculate_cagr(eps, num_years-1)
                    self._print_metric(f"EPS CAGR ({num_years} years):", eps_growth*100, "%")
                
                # Free cash flow growth
                if 'Free Cash Flow' in cf.index:
                    fcf_growth = self.calculate_cagr(cf.loc['Free Cash Flow'], num_years-1)
                    self._print_metric(f"FCF CAGR ({num_years} years):", fcf_growth*100, "%")
                
                # Equity growth
                if 'Total Equity Gross Minority Interest' in bs.index:
                    equity_growth = self.calculate_cagr(bs.loc['Total Equity Gross Minority Interest'], num_years-1)
                    self._print_metric(f"Equity CAGR ({num_years} years):", equity_growth*100, "%")
            else:
                print("\n  Insufficient data for trend analysis (need at least 2 years of data)")
                
            print(f"{'='*50}")
            
            return {
                "Revenue Growth (YoY)": rev_growth,
                "EPS Growth (YoY)": eps_growth,
                "FCF Growth (YoY)": fcf_growth,
                "Equity Growth (YoY)": equity_growth
            }

        except Exception as e:
            return f"Error in trend analysis: {str(e)}"

# Example usage
if __name__ == "__main__":
    ticker = "NVDA"
    peers = ["GOOGL", "AMZN", "AAPL"]  
    
    analyzer = FinancialAnalyzer(ticker)
    
    # Core financial statements
    analyzer.get_balance_sheet(plot=True)
    analyzer.get_income_statement(plot=True)
    analyzer.get_cash_flow(plot=True)
    
    # Ratios and metrics
    financial_ratios = analyzer.get_financial_ratios()
    print("\nFinancial Ratios:")
    for category, ratio_dict in financial_ratios.items():
        print(f"\n{category}:")
        for ratio, value in ratio_dict.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {ratio}: {value:.2f}" if ratio != "Interest Coverage" else f"  {ratio}: {value:.1f}")
                else:
                    print(f"  {ratio}: {value}")
            else:
                print(f"  {ratio}: N/A (data not available)")
    
    additional_metrics = analyzer.get_additional_metrics()
    print("\nAdditional Metrics:")
    for metric, value in additional_metrics.items():
        if value is not None:
            print(f"  {metric}: {value}")
        else:
            print(f"  {metric}: N/A (data not available)")
    
    # Growth and efficiency analysis
    growth_metrics = analyzer.get_growth_metrics()
    print("\nGrowth Metrics:")
    for metric, value in growth_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value*100:.2f}%")
        else:
            print(f"  {metric}: {value}")
    
    trend_analysis = analyzer.get_trend_analysis()
    print("\nTrend Analysis:")
    for metric, value in trend_analysis.items():
        if isinstance(value, float):
            print(f"  {metric}: {value*100:.2f}%")
        else:
            print(f"  {metric}: {value}")
    
    # Dividend and risk analysis
    dividend_analysis = analyzer.get_dividend_analysis()
    print("\nDividend Analysis:")
    for metric, value in dividend_analysis.items():
        print(f"  {metric}: {value}")
    
    risk_metrics = analyzer.get_risk_metrics()
    print("\nRisk Metrics:")
    for metric, value in risk_metrics.items():
        print(f"  {metric}: {value}")
    
 