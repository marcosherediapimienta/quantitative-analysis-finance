import yfinance as yf
import pandas as pd
from scripts.fundamental_analysis import BalanceSheetAnalyzer

# Uso de la clase
ticker = "AAPL" 
analyzer = BalanceSheetAnalyzer(ticker) 
analyzer.get_balance_sheet() 
analyzer = BalanceSheetAnalyzer(ticker)
analyzer.get_income_statement()