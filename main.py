import yfinance as yf
import pandas as pd
from scripts.fundamental_analysis import FinancialAnalyzer


ticker = "AAPL"
analyzer = FinancialAnalyzer(ticker)
balance_sheet = analyzer.get_balance_sheet()
income_statement = analyzer.get_income_statement()
cash_flow = analyzer.get_cash_flow()
