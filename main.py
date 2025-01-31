from scripts.fundamental_analysis import get_company_overview, get_etf_profile, get_dividends, get_splits, get_income_statement, get_balance_sheet, get_cash_flow, get_earnings, get_listing_status, get_earnings_calendar

# Ejemplo de uso
symbol = 'IBM'  # Puedes cambiar este valor por el ticker que desees

overview = get_company_overview(symbol)
#etf_profile = get_etf_profile('QQQ')  # Cambiar por el ticker del ETF que deseas
#dividends = get_dividends(symbol)
#splits = get_splits(symbol)
#income_statement = get_income_statement(symbol)
#balance_sheet = get_balance_sheet(symbol)
#cash_flow = get_cash_flow(symbol)
#earnings = get_earnings(symbol)
#listing_status = get_listing_status(state='active')  # Cambiar a 'delisted' si deseas activos deslistados
#earnings_calendar = get_earnings_calendar(symbol)

# Mostrar los resultados
print('Company Overview:', overview)
#print('ETF Profile:', etf_profile)
#print('Dividends:', dividends)
#print('Splits:', splits)
#print('Income Statement:', income_statement)
#print('Balance Sheet:', balance_sheet)
#print('Cash Flow:', cash_flow)
#print('Earnings:', earnings)
#print('Listing Status:', listing_status)
#print('Earnings Calendar:', earnings_calendar)