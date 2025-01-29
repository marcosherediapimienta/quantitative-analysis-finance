from scripts.fundamental_analysis import FundamentalAnalysis

tickers = ["MSFT"]
fa = FundamentalAnalysis(tickers)
df = fa.get_fundamental_data()
print(df)
