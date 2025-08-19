import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.Fundamental_Analysis import FinancialAnalyzer
from scripts.Technical_Analysis import TechnicalAnalysis
from scripts.Market_Risk_Analysis import RiskAnalysis

#Fundamental & Technical Analysis

#ticker = "NVDA"
#analyzer = FinancialAnalyzer(ticker)
#analyzer.get_balance_sheet(plot=True)
#analyzer.get_income_statement(plot=True)
#analyzer.get_cash_flow(plot=True)
#analyzer.get_financial_ratios()
#ta_analysis = TechnicalAnalysis(ticker)
#ta_analysis.plot_indicators()

#tickers = ['AAPL', 'GOOGL']
#weights = [0.6, 0.4]
#portfolio = RiskAnalysis(tickers, weights)
#portfolio.summary()
