import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.FundamentalAnalysis import FinancialAnalyzer
from scripts.TechnicalAnalysis import TechnicalAnalysis
from scripts.Market_Risk_Analysis import RiskAnalysis

ticker = "NVDA"
#analyzer = FinancialAnalyzer(ticker)
#analyzer.get_balance_sheet(plot=True)
#analyzer.get_income_statement(plot=True)
#analyzer.get_cash_flow(plot=True)
#analyzer.get_financial_ratios()

ta_analysis = TechnicalAnalysis(ticker)
ta_analysis.plot_indicators()

#RiskAnalysis
#tickers = ["NVDA", "META", "AMZN", "MSFT"]
#weights = [0.4, 0.4, 0.1, 0.1] 
#portfolio = RiskAnalysis(tickers, weights)
#portfolio.summary()
