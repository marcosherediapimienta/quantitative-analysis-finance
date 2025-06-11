import yfinance as yf
import pandas as pd
from finta import TA
import matplotlib.pyplot as plt
import mplfinance as mpf
import os

# Ensure the visualizations directory exists
visualizations_dir = os.path.join('quantitative-analysis-finance', 'portfolio_management', 'visualizations')
os.makedirs(visualizations_dir, exist_ok=True)

def descargar_datos(ticker='AAPL', interval='daily', start=None, end=None):
    """Descarga datos históricos de Yahoo Finance según el intervalo especificado."""
    print(f"Descargando datos para {ticker} con intervalo {interval}...")
    if start is None:
        start = pd.Timestamp.today() - pd.Timedelta(days=1)
    if end is None:
        end = pd.Timestamp.today()
    
    # Map the interval to yfinance's interval format
    yf_interval = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}.get(interval)
    if yf_interval is None:
        raise ValueError("Intervalo no soportado. Usa 'daily', 'weekly' o 'monthly'.")

    datos = yf.download(ticker, start=start, end=end, interval=yf_interval)
    datos = datos.reset_index()  # Reset index to ensure 'Date' is a column
    
    # Flatten MultiIndex if present
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = [' '.join(col).strip() for col in datos.columns]
    
    # Remove ticker suffix from column names
    datos.columns = [col.split(' ')[0].lower() for col in datos.columns]
    
    print("Datos descargados correctamente.")
    print(datos.head())  # Print the first few rows to verify
    
    return datos

def calcular_sma_multiple(df):
    """Calcula la SMA para los periodos de 20, 50 y 200 días."""
    print("\nCalculando SMA para periodos de 20, 50 y 200 días...")
    df['sma_20'] = TA.SMA(df, 20)
    df['sma_50'] = TA.SMA(df, 50)
    df['sma_200'] = TA.SMA(df, 200)
    print(df[['date', 'close', 'sma_20', 'sma_50', 'sma_200']].tail())  # Print the last few rows to verify

def calcular_ema_multiple(df):
    """Calcula la EMA para los periodos de 20 y 50 días."""
    print("\nCalculando EMA para periodos de 20 y 50 días...")
    df['ema_20'] = TA.EMA(df, 20)
    df['ema_50'] = TA.EMA(df, 50)
    print(df[['date', 'close', 'ema_20', 'ema_50']].tail())  # Print the last few rows to verify

def calcular_rsi(df, period=14):
    """Calcula el Índice de Fuerza Relativa (RSI)"""
    print("\nCalculando RSI...")
    df['rsi'] = TA.RSI(df, period)
    print(df[['date', 'close', 'rsi']].tail())  # Print the last few rows to verify

def calcular_macd(df):
    """Calcula el MACD y la señal"""
    print("\nCalculando MACD...")
    df['macd'] = TA.MACD(df)['MACD']
    df['signal_line'] = TA.MACD(df)['SIGNAL']
    print(df[['date', 'macd', 'signal_line']].tail())  # Print the last few rows to verify

def calcular_bollinger_bands(df, period=20, std_dev=2):
    """Calcula las Bandas de Bollinger"""
    print("\nCalculando Bandas de Bollinger...")
    df['sma'] = df['close'].rolling(window=period).mean()
    df['std_dev'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['sma'] + (df['std_dev'] * std_dev)
    df['lower_band'] = df['sma'] - (df['std_dev'] * std_dev)
    print(df[['date', 'close', 'upper_band', 'lower_band']].tail())  # Print the last few rows to verify

def calcular_momentum(df, period=10):
    """Calcula el indicador de Momentum"""
    print("\nCalculando Momentum...")
    df['momentum'] = df['close'] - df['close'].shift(period)
    print(df[['date', 'close', 'momentum']].tail())  # Print the last few rows to verify

def calcular_adx(df, period=14):
    """Calcula el Average Directional Index (ADX)."""
    print("\nCalculando ADX...")
    df['adx'] = TA.ADX(df, period)
    print(df[['date', 'adx']].tail())

def calcular_obv(df):
    """Calcula el On-Balance Volume (OBV)."""
    print("\nCalculando OBV...")
    df['obv'] = TA.OBV(df)
    print(df[['date', 'obv']].tail())

def calcular_stochastic_oscillator(df, k_period=14, d_period=3):
    """Calcula el Stochastic Oscillator."""
    print("\nCalculando Stochastic Oscillator...")
    df['stoch_k'] = TA.STOCH(df, k_period)
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    print(df[['date', 'stoch_k', 'stoch_d']].tail())

# Plotting functions

def plot_candlestick_and_momentum(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 15), sharex=True)

    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot Momentum
    ax2.set_title(f'{ticker} - Momentum')
    ax2.plot(df['date'], df['momentum'], label='Momentum', color='orange')
    ax2.axhline(0, color='black', linestyle='--')  # Zero line
    ax2.set_ylabel('Momentum')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    ax2.legend(loc='best')

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_candlestick_and_momentum.png'))
    plt.close(fig)
    return fig

def plot_candlestick_and_rsi(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 15), sharex=True)

    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot RSI
    ax2.plot(df['date'], df['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--')  # Overbought line
    ax2.axhline(30, color='green', linestyle='--')  # Oversold line
    ax2.fill_between(df['date'], 30, 70, color='purple', alpha=0.1)
    ax2.set_title(f'{ticker} - RSI')
    ax2.set_ylabel('RSI')
    ax2.grid(True)
    ax2.legend(loc='best')

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_candlestick_and_rsi.png'))
    return fig

def plot_candlestick_and_macd(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 15), sharex=True)

    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot MACD
    ax2.plot(df['date'], df['macd'], label='MACD', color='blue')
    ax2.plot(df['date'], df['signal_line'], label='Signal Line', color='red')
    ax2.bar(df['date'], df['macd'] - df['signal_line'], label='MACD Histogram', color='gray', alpha=0.3)
    ax2.set_title(f'{ticker} - MACD')
    ax2.set_ylabel('MACD')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    ax2.legend(loc='best')

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_candlestick_and_macd.png'))
    return fig

def plot_candlestick_and_bollinger(df, ticker):
    fig, ax1 = plt.subplots(figsize=(14, 10))

    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart with Bollinger Bands')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot Bollinger Bands
    ax1.plot(df['date'], df['upper_band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(df['date'], df['lower_band'], label='Lower Band', color='green', linestyle='--')
    ax1.fill_between(df['date'], df['lower_band'], df['upper_band'], color='gray', alpha=0.1)

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_candlestick_and_bollinger.png'))
    plt.close(fig)
    return fig

def plot_sma_multiple(df, ticker):
    """Plot SMA for 20, 50, and 200 days with candlestick chart and save as PNG."""
    fig, ax1 = plt.subplots(figsize=(14, 10))
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.plot(df['date'], df['sma_20'], label='SMA 20', color='red', linestyle='--')
    ax1.plot(df['date'], df['sma_50'], label='SMA 50', color='green', linestyle='--')
    ax1.plot(df['date'], df['sma_200'], label='SMA 200', color='orange', linestyle='--')
    ax1.set_title(f'{ticker} - Candlestick Chart and SMAs')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_sma_multiple.png'))
    plt.close(fig)
    return fig

def plot_ema_multiple(df, ticker):
    """Plot EMA for 20 and 50 days with candlestick chart and save as PNG."""
    fig, ax1 = plt.subplots(figsize=(14, 10))
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.plot(df['date'], df['ema_20'], label='EMA 20', color='purple', linestyle='--')
    ax1.plot(df['date'], df['ema_50'], label='EMA 50', color='brown', linestyle='--')
    ax1.set_title(f'{ticker} - Candlestick Chart and EMAs')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_ema_multiple.png'))
    plt.close(fig)
    return fig

def plot_adx(df, ticker):
    """Plot candlestick chart and ADX below, then save as PNG."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 15), sharex=True)
    
    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot ADX
    ax2.plot(df['date'], df['adx'], label='ADX', color='blue')
    ax2.set_title(f'{ticker} - ADX')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('ADX')
    ax2.legend(loc='best')
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_adx.png'))
    plt.close(fig)
    return fig

def plot_stochastic_oscillator(df, ticker):
    """Plot candlestick chart and Stochastic Oscillator below, then save as PNG."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 15), sharex=True)
    
    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot Stochastic Oscillator
    ax2.plot(df['date'], df['stoch_k'], label='%K', color='blue')
    ax2.plot(df['date'], df['stoch_d'], label='%D', color='red')
    ax2.axhline(80, color='red', linestyle='--')  # Overbought line
    ax2.axhline(20, color='green', linestyle='--')  # Oversold line
    ax2.set_title(f'{ticker} - Stochastic Oscillator')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Stochastic')
    ax2.legend(loc='best')
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_stochastic_oscillator.png'))
    plt.close(fig)
    return fig

def plot_macd_with_adx(df, ticker):
    """Plot candlestick chart with MACD and ADX below, then save as PNG."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot MACD
    ax2.plot(df['date'], df['macd'], label='MACD', color='blue')
    ax2.plot(df['date'], df['signal_line'], label='Signal Line', color='red')
    ax2.bar(df['date'], df['macd'] - df['signal_line'], label='MACD Histogram', color='gray', alpha=0.3)
    ax2.set_title(f'{ticker} - MACD')
    ax2.set_ylabel('MACD')
    ax2.grid(True)
    ax2.legend(loc='best')

    # Plot ADX
    ax3.plot(df['date'], df['adx'], label='ADX', color='green')
    ax3.set_title(f'{ticker} - ADX')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('ADX')
    ax3.legend(loc='best')
    ax3.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_macd_with_adx.png'))
    plt.close(fig)
    return fig

def plot_macd_with_stochastic(df, ticker):
    """Plot candlestick chart with MACD and Stochastic Oscillator below, then save as PNG."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot MACD
    ax2.plot(df['date'], df['macd'], label='MACD', color='blue')
    ax2.plot(df['date'], df['signal_line'], label='Signal Line', color='red')
    ax2.bar(df['date'], df['macd'] - df['signal_line'], label='MACD Histogram', color='gray', alpha=0.3)
    ax2.set_title(f'{ticker} - MACD')
    ax2.set_ylabel('MACD')
    ax2.grid(True)
    ax2.legend(loc='best')

    # Plot Stochastic Oscillator
    ax3.plot(df['date'], df['stoch_k'], label='%K', color='blue')
    ax3.plot(df['date'], df['stoch_d'], label='%D', color='red')
    ax3.axhline(80, color='red', linestyle='--')  # Overbought line
    ax3.axhline(20, color='green', linestyle='--')  # Oversold line
    ax3.set_title(f'{ticker} - Stochastic Oscillator')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Stochastic')
    ax3.legend(loc='best')
    ax3.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_macd_with_stochastic.png'))
    plt.close(fig)
    return fig

def plot_rsi_with_adx(df, ticker):
    """Plot candlestick chart with RSI and ADX below, then save as PNG."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot RSI
    ax2.plot(df['date'], df['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--')  # Overbought line
    ax2.axhline(30, color='green', linestyle='--')  # Oversold line
    ax2.fill_between(df['date'], 30, 70, color='purple', alpha=0.1)
    ax2.set_title(f'{ticker} - RSI')
    ax2.set_ylabel('RSI')
    ax2.grid(True)

    # Plot ADX
    ax3.plot(df['date'], df['adx'], label='ADX', color='green')
    ax3.set_title(f'{ticker} - ADX')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('ADX')
    ax3.legend(loc='best')
    ax3.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_rsi_with_adx.png'))
    plt.close(fig)
    return fig

def plot_rsi_with_stochastic(df, ticker):
    """Plot candlestick chart with RSI and Stochastic Oscillator below, then save as PNG."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    # Plot candlestick chart with volume
    mpf.plot(df.set_index('date'), type='candle', ax=ax1, show_nontrading=True, style='yahoo')
    ax1.set_title(f'{ticker} - Candlestick Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot volume
    ax1v = ax1.twinx()
    ax1v.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
    ax1v.set_ylabel('Volume')
    ax1v.set_ylim(0, df['volume'].max() * 4)
    ax1v.grid(False)

    # Plot RSI
    ax2.plot(df['date'], df['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--')  # Overbought line
    ax2.axhline(30, color='green', linestyle='--')  # Oversold line
    ax2.fill_between(df['date'], 30, 70, color='purple', alpha=0.1)
    ax2.set_title(f'{ticker} - RSI')
    ax2.set_ylabel('RSI')
    ax2.grid(True)

    # Plot Stochastic Oscillator
    ax3.plot(df['date'], df['stoch_k'], label='%K', color='blue')
    ax3.plot(df['date'], df['stoch_d'], label='%D', color='red')
    ax3.axhline(80, color='red', linestyle='--')  # Overbought line
    ax3.axhline(20, color='green', linestyle='--')  # Oversold line
    ax3.set_title(f'{ticker} - Stochastic Oscillator')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Stochastic')
    ax3.legend(loc='best')
    ax3.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(visualizations_dir, f'{ticker}_rsi_with_stochastic.png'))
    plt.close(fig)
    return fig

if __name__ == "__main__":
    ticker = 'META'  # Default ticker
    start_date = '2022-01-01'  # Default start date
    end_date = '2023-01-01'  # Default end date
    interval = 'weekly'  # Default interval
    datos = descargar_datos(ticker, start=start_date, end=end_date, interval=interval)

    # Calculate indicators
    calcular_sma_multiple(datos)
    calcular_ema_multiple(datos)
    calcular_rsi(datos)
    calcular_macd(datos)
    calcular_bollinger_bands(datos)
    calcular_momentum(datos)
    calcular_adx(datos)
    calcular_obv(datos)
    calcular_stochastic_oscillator(datos)

    # Plot indicators
    plot_sma_multiple(datos, ticker)
    plot_ema_multiple(datos, ticker)
    plot_candlestick_and_momentum(datos, ticker)
    plot_candlestick_and_rsi(datos, ticker)
    plot_candlestick_and_macd(datos, ticker)
    plot_candlestick_and_bollinger(datos, ticker)
    plot_adx(datos, ticker)
    plot_stochastic_oscillator(datos, ticker)

    # Plot combined indicators
    plot_macd_with_adx(datos, ticker)
    plot_macd_with_stochastic(datos, ticker)
    plot_rsi_with_adx(datos, ticker)
    plot_rsi_with_stochastic(datos, ticker)

