import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.FundamentalAnalysis import FinancialAnalyzer
from scripts.TechnicalAnalysis import TechnicalAnalysis

ticker = "META"
#analyzer = FinancialAnalyzer(ticker)
#analyzer.get_balance_sheet(plot=True)
#analyzer.get_income_statement(plot=True)
#analyzer.get_cash_flow(plot=True)
#analyzer.get_financial_ratios()

#plt.show()

ta_analysis = TechnicalAnalysis(ticker)
ta_analysis.plot_indicators()