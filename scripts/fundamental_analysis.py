import yfinance as yf
import pandas as pd

class FundamentalAnalysis:
    def __init__(self, tickers):
        """
        Inicializa la clase con una lista de tickers.
        :param tickers: Lista de tickers (ejemplo: ["AAPL", "MSFT"])
        """
        self.tickers = tickers
    
    def get_fundamental_data(self):
        """
        Obtiene los datos fundamentales de los tickers especificados.
        Devuelve un DataFrame con métricas clave.
        """
        fundamental_data = []
        
        for ticker in self.tickers:
            print(f"Downloading data for {ticker}...")
            try:
                stock = yf.Ticker(ticker)
                print(f"Getting info for ticker {ticker}...")
                
                # Obtener información clave
                info = stock.info
                financials = stock.financials
                balance_sheet = stock.balance_sheet
                cashflow = stock.cashflow

                # Mostrar las primeras filas de cada dataframe para depuración
                print("Financials:\n", financials.head())
                print("Balance Sheet:\n", balance_sheet.head())
                print("Cashflow:\n", cashflow.head())
                
                # Extraer datos relevantes
                data = {
                    "Ticker": ticker,
                    "Sector": info.get("sector", "N/A"),
                    "Industry": info.get("industry", "N/A"),
                    "Market Cap": info.get("marketCap", "N/A"),
                    "PE Ratio": info.get("trailingPE", "N/A"),
                    "PB Ratio": info.get("priceToBook", "N/A"),
                    "EPS": info.get("trailingEps", "N/A"),
                    "Dividend Yield": info.get("dividendYield", "N/A"),
                    "Revenue (TTM)": self.get_data(financials, "Total Revenue"),
                    "Net Income (TTM)": self.get_data(financials, "Net Income"),
                    "Total Assets": self.get_data(balance_sheet, "Total Assets"),
                    "Total Liabilities": self.get_data(balance_sheet, "Total Liabilities Net Minority Interest"),
                    "Free Cash Flow (TTM)": self.get_data(cashflow, "Free Cash Flow"),
                }

                fundamental_data.append(data)

            except Exception as e:
                print(f"Error obteniendo datos para {ticker}: {e}")

        # Convertir a DataFrame
        df_fundamental = pd.DataFrame(fundamental_data)

        # Formatear números grandes para mayor legibilidad
        for column in ["Market Cap", "Revenue (TTM)", "Net Income (TTM)", "Total Assets", "Total Liabilities", "Free Cash Flow (TTM)"]:
            if column in df_fundamental.columns:
                df_fundamental[column] = pd.to_numeric(df_fundamental[column], errors="coerce")  # Convertir a numérico
                df_fundamental[column] = df_fundamental[column].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        
        return df_fundamental
    
    def get_data(self, df, column):
        """
        Verifica si la columna está en el dataframe y extrae el valor.
        :param df: El dataframe de financials, balance_sheet o cashflow.
        :param column: El nombre de la columna a extraer.
        :return: El valor de la columna o "N/A" si no existe.
        """
        if column in df.index:
            return df.loc[column].values[0]
        else:
            return "N/A"

