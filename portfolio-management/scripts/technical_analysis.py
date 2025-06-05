import yfinance as yf
import pandas as pd
from finta import TA
import matplotlib.pyplot as plt


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
    """Calcula la media móvil simple (SMA) y las señales de compra/venta usando finta"""
    print("\nCalculando SMA...")
    df['sma_20'] = TA.SMA(df, period)
    # Calculate buy and sell signals
    df['signal'] = 0
    df.loc[df['close'] > df['sma_20'], 'signal'] = 1  # Buy signal
    df.loc[df['close'] < df['sma_20'], 'signal'] = -1  # Sell signal
    df['position'] = df['signal'].diff()
    print(df[['date', 'close', 'sma_20', 'position']].tail())  # Print the last few rows to verify

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

    # Plot buy signals
    buy_signals = df[df['position'] == 2]
    plt.scatter(buy_signals['date'], buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100, zorder=5)

    # Plot sell signals
    sell_signals = df[df['position'] == -2]
    plt.scatter(sell_signals['date'], sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100, zorder=5)

    plt.savefig(f'/home/marcos/Escritorio/mhp/quantitative-analysis-finance/portfolio-management/visualizations/{ticker}_sma_plot.png')  

def calcular_rsi(df, period=14):
    """Calcula el Índice de Fuerza Relativa (RSI)"""
    print("\nCalculando RSI...")
    df['rsi'] = TA.RSI(df, period)
    print(df[['date', 'close', 'rsi']].tail())  # Print the last few rows to verify

def plot_price_and_rsi_separately(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot close price
    ax1.set_title(f'{ticker} - Close Price')
    ax1.plot(df['date'], df['close'], label='Close Price', color='blue')
    ax1.set_ylabel('Close Price')
    ax1.grid(True)

    # Plot RSI
    ax2.set_title(f'{ticker} - RSI')
    ax2.plot(df['date'], df['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--')  # Overbought line
    ax2.axhline(30, color='green', linestyle='--')  # Oversold line
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    # Plot RSI with shaded area
    ax2.fill_between(df['date'], 30, 70, color='purple', alpha=0.1)  # Shade the RSI area between 30 and 70

    plt.tight_layout()
    plt.savefig(f'/home/marcos/Escritorio/mhp/quantitative-analysis-finance/portfolio-management/visualizations/{ticker}_price_rsi_separate_plot.png')  # Save the separate plots

def calcular_macd(df):
    """Calcula el MACD y la señal"""
    print("\nCalculando MACD...")
    df['macd'] = TA.MACD(df)['MACD']
    df['signal_line'] = TA.MACD(df)['SIGNAL']
    print(df[['date', 'macd', 'signal_line']].tail())  # Print the last few rows to verify

def plot_price_and_macd_with_histogram(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot close price
    ax1.set_title(f'{ticker} - Close Price')
    ax1.plot(df['date'], df['close'], label='Close Price', color='blue')
    ax1.set_ylabel('Close Price')
    ax1.grid(True)

    # Plot MACD and Histogram
    ax2.set_title(f'{ticker} - MACD')
    ax2.plot(df['date'], df['macd'], label='MACD', color='blue')
    ax2.plot(df['date'], df['signal_line'], label='Signal Line', color='red')
    ax2.bar(df['date'], df['macd'] - df['signal_line'], label='MACD Histogram', color='gray', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f'/home/marcos/Escritorio/mhp/quantitative-analysis-finance/portfolio-management/visualizations/{ticker}_price_macd_combined_plot.png')  # Save the combined plot

def calcular_bollinger_bands(df, period=20, std_dev=2):
    """Calcula las Bandas de Bollinger"""
    print("\nCalculando Bandas de Bollinger...")
    df['sma'] = df['close'].rolling(window=period).mean()
    df['std_dev'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['sma'] + (df['std_dev'] * std_dev)
    df['lower_band'] = df['sma'] - (df['std_dev'] * std_dev)
    print(df[['date', 'close', 'upper_band', 'lower_band']].tail())  # Print the last few rows to verify

# Function to plot Bollinger Bands with buy/sell signals
def plot_bollinger_bands(df, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(df['date'], df['close'], label='Close Price', color='blue')
    plt.plot(df['date'], df['upper_band'], label='Upper Band', color='red', linestyle='--')
    plt.plot(df['date'], df['lower_band'], label='Lower Band', color='green', linestyle='--')
    plt.fill_between(df['date'], df['lower_band'], df['upper_band'], color='gray', alpha=0.1)

    # Plot buy signals
    buy_signals = df[df['close'] < df['lower_band']]
    plt.scatter(buy_signals['date'], buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100, zorder=5)

    # Plot sell signals
    sell_signals = df[df['close'] > df['upper_band']]
    plt.scatter(sell_signals['date'], sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100, zorder=5)

    plt.title(f'{ticker} - Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'/home/marcos/Escritorio/mhp/quantitative-analysis-finance/portfolio-management/visualizations/{ticker}_bollinger_bands_plot.png')  # Save the Bollinger Bands plot

# Example usage
ticker = "AAPL"
datos = descargar_datos(ticker)
calcular_sma(datos)
plot_sma(datos, ticker)
calcular_rsi(datos)
plot_price_and_rsi_separately(datos, ticker)
calcular_macd(datos)
plot_price_and_macd_with_histogram(datos, ticker)
calcular_bollinger_bands(datos)
plot_bollinger_bands(datos, ticker)
