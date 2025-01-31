import requests

# Usando tu clave API
api_key = 'T7N2JJGEJG6MJ9OZ'

# Función para obtener los datos fundamentales de la empresa
def get_company_overview(symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Función para obtener el perfil de un ETF
def get_etf_profile(symbol):
    url = f'https://www.alphavantage.co/query?function=ETF_PROFILE&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Función para obtener los dividendos de la empresa
def get_dividends(symbol):
    url = f'https://www.alphavantage.co/query?function=DIVIDENDS&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Función para obtener los splits de la empresa
def get_splits(symbol):
    url = f'https://www.alphavantage.co/query?function=SPLITS&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Función para obtener el estado de resultados de la empresa
def get_income_statement(symbol):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Función para obtener el balance de la empresa
def get_balance_sheet(symbol):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Función para obtener el flujo de caja de la empresa
def get_cash_flow(symbol):
    url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Función para obtener las ganancias de la empresa
def get_earnings(symbol):
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Función para obtener el estado de listado y exclusión de una acción o ETF
def get_listing_status(state='active', date=None):
    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}'
    if date:
        url += f'&date={date}'
    if state:
        url += f'&state={state}'
    response = requests.get(url)
    data = response.content.decode('utf-8')
    return data

def get_earnings_calendar(symbol=None, horizon='3month'):
    url = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={api_key}'
    if symbol:
        url += f'&symbol={symbol}'
    if horizon:
        url += f'&horizon={horizon}'
    response = requests.get(url)

    # Imprimir la respuesta para depuración
    print(response.text)  # Esto imprimirá el contenido completo de la respuesta

    try:
        data = response.json()  # Intentar decodificar la respuesta JSON
    except ValueError:
        print("Error al decodificar la respuesta JSON")
        return None

    return data