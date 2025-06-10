import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FinancialAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
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

                # Save the figure as a .png file
                plt.savefig(f'quantitative-analysis-finance/portfolio_management/visualizations/{title}_{category}_plot.png')

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

            print(f"\n{'='*50}")
            print(f"Balance Sheet for {self.ticker}:")
            for category, accounts in available_accounts.items():
                if accounts:
                    print(f"\n{category}:")
                    for account in accounts:
                        values = info.loc[account]
                        self._print_values(account, values)

            if plot:
                figures = self._plot_metrics(info, balance_sheet, "Balance Sheet")
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
                "Revenue": ['Total Revenue', 'Cost Of Revenue'],
                "Gross and Operating Profitability": ['Gross Profit', 'Operating Income'],
                "Operating Profit and Pre-Tax": ['EBITDA', 'Earnings Before Interest and Taxes (EBIT)', 'Earnings Before Taxes (EBT)'],
                "Financial Profitability": ['Net Interest Income', 'Net Income']
            }

            available_metrics = self._filter_metrics(info, income_stmt)

            print(f"\n{'='*50}")
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

            print(f"\n{'='*50}")
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
                    "Debt-to-Equity Ratio": info.get('debtToEquity')
                }
            }

            print(f"\n{'='*50}")
            print(f"Financial Ratios for {self.ticker}:")
            for category, ratio_dict in ratios.items():
                print(f"\n{category}:")
                for ratio, value in ratio_dict.items():
                    if value is not None:
                        if isinstance(value, float):
                            print(f"  {ratio}: {value:.2f}" if ratio != "Interest Coverage" else f"  {ratio}: {value:.1f}")
                        else:
                            print(f"  {ratio}: {value}")
                    else:
                        print(f"  {ratio}: N/A (data not available)")

        except Exception as e:
            print(f"Error retrieving financial ratios for {self.ticker}: {str(e)}")

    def get_dividend_analysis(self):
        """Analyze dividend payments and yield"""
        try:
            div_info = self.company.dividends
            info = self.company.info
            
            print(f"\n{'='*50}")
            print(f"Dividend Analysis for {self.ticker}:")
            
            # Debugging: Print raw dividend yield data
            current_yield = info.get('dividendYield', 'N/A')

            # Correct the calculation of the Current Dividend Yield
            print(f"\n  Current Dividend Yield: {current_yield if isinstance(current_yield, float) else 'N/A'}%")
            
            # Basic data
            payout_ratio = info.get('payoutRatio', 'N/A')
            five_year_growth = info.get('fiveYearAvgDividendYield', 'N/A')
            
            print(f"\n  Payout Ratio: {payout_ratio*100 if isinstance(payout_ratio, float) else 'N/A'}%")
            
            # Dividend history
            if not div_info.empty:
                print("\n  Dividend History:")
                div_history = div_info.tail(5)  # Last 5 dividends
                for date, amount in div_history.items():
                    print(f"    {date.strftime('%Y-%m-%d')}: ${amount:.4f}")
                
                # Dividend plot
                plt.figure(figsize=(10, 5))
                div_info.plot(title=f"{self.ticker} Dividend History")
                plt.ylabel('Dividend Amount ($)')
                plt.grid(True)
                plt.savefig('quantitative-analysis-finance/portfolio_management/visualizations/dividend_history.png')
            else:
                print("\n  No dividend history available")
                
        except Exception as e:
            print(f"Error retrieving dividend info: {str(e)}")

    def get_growth_metrics(self):
        """Analyze growth metrics"""
        try:
            info = self.company.info
            
            growth_metrics = {
                "Revenue Growth (YoY)": info.get('revenueGrowth', 'N/A'),
                "Earnings Growth (YoY)": info.get('earningsGrowth', 'N/A'),
                "Next 5 Years Growth Estimate": info.get('earningsQuarterlyGrowth', 'N/A')
            }
            
            print(f"\n{'='*50}")
            print(f"Growth Metrics for {self.ticker}:")
            for metric, value in growth_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value*100:.2f}%")
                else:
                    print(f"  {metric}: {value}")
                    
        except Exception as e:
            print(f"Error retrieving growth metrics: {str(e)}")

    def compare_with_peers(self, peers):
        """Compare key metrics with peer companies"""
        try:
            print(f"\n{'='*50}")
            print(f"Competitive Analysis for {self.ticker} vs Peers:")
            
            metrics = ['trailingPE', 'priceToBook', 'returnOnEquity', 
                      'debtToEquity', 'operatingMargins', 'currentRatio',
                      'grossMargins', 'dividendYield']
            
            comparison = pd.DataFrame(index=metrics)
            current_info = self.company.info
            comparison[self.ticker] = [current_info.get(m, None) for m in metrics]
            
            for peer in peers:
                try:
                    peer_info = yf.Ticker(peer).info
                    comparison[peer] = [peer_info.get(m, None) for m in metrics]
                except Exception as e:
                    print(f"  Warning: Could not get data for {peer}: {str(e)}")
                    comparison[peer] = [None]*len(metrics)
            
            # Clean data - replace None with 'N/A' for display
            display_comparison = comparison.fillna('N/A')
            print("\n" + str(display_comparison))
            
            # Plot only metrics with numeric data
            for metric in metrics:
                if pd.api.types.is_numeric_dtype(comparison.loc[metric]):
                    plt.figure(figsize=(10, 5))
                    comparison.loc[metric].plot(kind='bar', title=metric)
                    plt.ylabel(metric)
                    plt.grid(True)
                    plt.savefig(f'quantitative-analysis-finance/portfolio_management/visualizations/peer_comparison_{metric}.png')
                else:
                    print(f"  Cannot plot {metric} - no numeric data available")
                    
        except Exception as e:
            print(f"Error in peer comparison: {str(e)}")

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
            
            print(f"\n{'='*50}")
            print(f"Risk Metrics for {self.ticker}:")
            for metric, value in risk_metrics.items():
                print(f"  {metric}: {value}")
                
        except Exception as e:
            print(f"Error retrieving risk metrics: {str(e)}")

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
            
            print(f"\n{'='*50}")
            print(f"Additional Metrics for {self.ticker}:")
            for metric, value in additional_metrics.items():
                if value is not None:
                    print(f"  {metric}: {value}")
                else:
                    print(f"  {metric}: N/A (data not available)")
                    
        except Exception as e:
            print(f"Error retrieving additional metrics for {self.ticker}: {str(e)}")

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
            
            print(f"\n{'='*50}")
            print(f"Trend Analysis for {self.ticker}:")
            
            # Calculate number of years available
            num_years = len(bs.columns) if not bs.empty else 0
            
            if num_years >= 2:
                # Revenue growth
                if 'Total Revenue' in is_.index:
                    rev_growth = self.calculate_cagr(is_.loc['Total Revenue'], num_years)
                    print(f"\n  Revenue CAGR ({num_years} years): {rev_growth*100:.2f}%" if isinstance(rev_growth, float) else f"  Revenue CAGR: {rev_growth}")
                
                # EPS growth
                if 'Net Income' in is_.index and 'sharesOutstanding' in self.company.info:
                    net_income = is_.loc['Net Income']
                    shares = self.company.info['sharesOutstanding']
                    eps = net_income / shares
                    eps_growth = self.calculate_cagr(eps, num_years-1)
                    print(f"  EPS CAGR ({num_years} years): {eps_growth*100:.2f}%" if isinstance(eps_growth, float) else f"  EPS CAGR: {eps_growth}")
                
                # Free cash flow growth
                if 'Free Cash Flow' in cf.index:
                    fcf_growth = self.calculate_cagr(cf.loc['Free Cash Flow'], num_years-1)
                    print(f"  FCF CAGR ({num_years} years): {fcf_growth*100:.2f}%" if isinstance(fcf_growth, float) else f"  FCF CAGR: {fcf_growth}")
                
                # Equity growth
                if 'Total Equity Gross Minority Interest' in bs.index:
                    equity_growth = self.calculate_cagr(bs.loc['Total Equity Gross Minority Interest'], num_years-1)
                    print(f"  Equity CAGR ({num_years} years): {equity_growth*100:.2f}%" if isinstance(equity_growth, float) else f"  Equity CAGR: {equity_growth}")
            else:
                print("\n  Insufficient data for trend analysis (need at least 2 years of data)")
                
        except Exception as e:
            print(f"Error in trend analysis: {str(e)}")

# Example usage
if __name__ == "__main__":
    ticker = "META"
    peers = ["GOOGL", "AMZN", "AAPL"]  
    
    analyzer = FinancialAnalyzer(ticker)
    
    # Core financial statements
    analyzer.get_balance_sheet(plot=True)
    analyzer.get_income_statement(plot=True)
    analyzer.get_cash_flow(plot=True)
    
    # Ratios and metrics
    analyzer.get_financial_ratios()
    analyzer.get_additional_metrics()
    
    # Growth and efficiency analysis
    analyzer.get_growth_metrics()
    analyzer.get_trend_analysis()
    
    # Dividend and risk analysis
    analyzer.get_dividend_analysis()
    analyzer.get_risk_metrics()
    
    # Peer comparison
    analyzer.compare_with_peers(peers)
