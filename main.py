import yfinance as yf
import pandas as pd
from scripts.fundamental_analysis import FinancialAnalyzer

# Uso de la clase
ticker = "AAPL" 
analyzer = FinancialAnalyzer(ticker) 
analyzer.get_balance_sheet() 
analyzer.get_income_statement()
analyzer.get_cash_flow()    
