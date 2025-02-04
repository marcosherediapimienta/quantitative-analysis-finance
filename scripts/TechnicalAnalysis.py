import yfinance as yf
import pandas as pd
import datetime
import pandas_ta as ta
import matplotlib.pyplot as plt

class TechnicalAnalysis:
    def __init__(self, ticker, start="2024-02-04", end=None):
        if end is None:
            end = datetime.date.today().strftime("%Y-%m-%d")  
        self.ticker = ticker
        self.data = yf.download(ticker, start=start, end=end)
        self.add_indicators()

    def add_indicators(self):
        """Adds technical indicators to the dataframe, including volume."""
        self.data["Momentum"] = ta.mom(self.data["Close"], length=10)
        macd = ta.macd(self.data["Close"])
        self.data["MACD"] = macd["MACD_12_26_9"]
        self.data["MACD_Signal"] = macd["MACDs_12_26_9"]
        self.data["RSI"] = ta.rsi(self.data["Close"], length=14)
        
        # Stochastic Oscillator
        stoch = ta.stoch(self.data["High"], self.data["Low"], self.data["Close"])
        self.data["Stochastic_K"] = stoch["STOCHk_14_3_3"]
        self.data["Stochastic_D"] = stoch["STOCHd_14_3_3"]

        # ADX & DMI
        adx = ta.adx(self.data["High"], self.data["Low"], self.data["Close"])
        self.data["ADX"] = adx["ADX_14"]
        self.data["DMI_Plus"] = adx["DMP_14"]
        self.data["DMI_Minus"] = adx["DMN_14"]

        # Bollinger Bands
        bbands = ta.bbands(self.data["Close"], length=20)
        self.data["Bollinger_Upper"] = bbands["BBU_20_2.0"]
        self.data["Bollinger_Lower"] = bbands["BBL_20_2.0"]

    def plot_indicators(self):
        """Plots the technical indicators with correctly scaled volume in the MACD plot."""
        fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

        # Close Price (Serie temporal)
        axes[0].plot(self.data.index, self.data["Close"], label="Close Price", color="black")
        axes[0].set_title(f"{self.ticker} - Close Price")
        axes[0].legend()

        # Price and Bollinger Bands
        axes[1].plot(self.data.index, self.data["Close"], label="Close Price", color="black")
        axes[1].plot(self.data.index, self.data["Bollinger_Upper"], label="Bollinger Upper", linestyle="dashed", color="blue")
        axes[1].plot(self.data.index, self.data["Bollinger_Lower"], label="Bollinger Lower", linestyle="dashed", color="red")
        axes[1].set_title(f"{self.ticker} - Close Price & Bollinger Bands")
        axes[1].legend()

        # MACD with correctly scaled Volume
        ax_macd = axes[2]
        ax_vol = ax_macd.twinx()  # Eje secundario para el volumen
        ax_vol.fill_between(self.data.index, self.data["Volume"], color="gray", alpha=0.3, label="Volume")
        ax_macd.plot(self.data.index, self.data["MACD"], label="MACD", color="green", linewidth=1.5)
        ax_macd.plot(self.data.index, self.data["MACD_Signal"], label="MACD Signal", color="red", linewidth=1.5)

        ax_macd.set_title("MACD & Signal Line with Volume")
        ax_macd.legend(loc="upper left")
        ax_vol.legend(loc="upper right")

        ax_macd.set_ylabel("MACD")
        ax_vol.set_ylabel("Volume")

        # RSI
        axes[3].plot(self.data.index, self.data["RSI"], label="RSI", color="purple")
        axes[3].axhline(70, linestyle="dashed", color="red")
        axes[3].axhline(30, linestyle="dashed", color="green")
        axes[3].set_title("Relative Strength Index (RSI)")
        axes[3].legend()

        # ADX & DMI
        axes[4].plot(self.data.index, self.data["ADX"], label="ADX", color="black")
        axes[4].plot(self.data.index, self.data["DMI_Plus"], label="+DI", color="green")
        axes[4].plot(self.data.index, self.data["DMI_Minus"], label="-DI", color="red")
        axes[4].set_title("ADX & DMI")
        axes[4].legend()

        # Stochastic Oscillator
        axes[5].plot(self.data.index, self.data["Stochastic_K"], label="Stochastic %K", color="blue")
        axes[5].plot(self.data.index, self.data["Stochastic_D"], label="Stochastic %D", color="orange")
        axes[5].axhline(80, linestyle="dashed", color="red")  # Overbought
        axes[5].axhline(20, linestyle="dashed", color="green")  # Oversold
        axes[5].set_title("Stochastic Oscillator")
        axes[5].legend()

        plt.tight_layout()
        plt.show()
