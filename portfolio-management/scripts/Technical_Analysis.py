import yfinance as yf
import pandas as pd
from finta import TA
import matplotlib.pyplot as plt
import matplotlib


def descargar_datos(ticker, start="2024-02-04", end="2025-02-04"):
    """Descarga datos históricos de Yahoo Finance"""
    print(f"Descargando datos para {ticker} desde {start} hasta {end}...")
    datos = yf.download(ticker, start=start, end=end)
    datos = datos.reset_index()  # Reset index to ensure 'Date' is a column
    
    # Flatten MultiIndex if present
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = [' '.join(col).strip() for col in datos.columns]
    
    # Remove ticker suffix from column names
    datos.columns = [col.split(' ')[0].lower() for col in datos.columns]
    
    print("Datos descargados correctamente.")
    print(datos.head())  # Print the first few rows to verify
    return datos

def calcular_sma(df, period=20):
    """Calcula la media móvil simple (SMA) usando finta"""
    print("\nCalculando SMA...")
    df['sma_20'] = TA.SMA(df, period)
    print(df[['date', 'close', 'sma_20']].tail())  # Print the last few rows to verify

# Function to plot and save the SMA
def plot_sma(df, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(df['date'], df['close'], label='Close Price', color='blue')
    plt.plot(df['date'], df['sma_20'], label='SMA 20', color='red', linestyle='--')
    plt.title(f'{ticker} - Close Price and SMA 20')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'{ticker}_sma_plot.png')  # Save the plot as a PNG file
    # plt.show()  # Comment out the interactive show function

#
# Example usage
ticker = "AAPL"
datos = descargar_datos(ticker)
calcular_sma(datos)
plot_sma(datos, ticker)
