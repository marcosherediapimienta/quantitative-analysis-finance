"""
data_options.py

This script allows you to:
  1. Retrieve all available option expiration dates for a given Yahoo Finance ticker.
  2. For a selected expiration date, fetch and display the calls and puts tables.
"""

import yfinance as yf
import pandas as pd


def get_expirations(ticker: str) -> list:
    """
    Returns the list of available option expiration dates for the ticker.
    """
    tk = yf.Ticker(ticker)
    return tk.options


def get_option_chain(ticker: str, expiration: str) -> tuple:
    """
    Returns two DataFrames: calls and puts for the given ticker and expiration date.
    """
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiration)
    return chain.calls, chain.puts


def get_us_risk_free_rate() -> float:
    """
    Fetches the current US 10-year Treasury yield via Yahoo Finance ticker ^TNX.
    Returns the rate as a decimal (e.g. 0.0435 for 4.35%) or None if unavailable.
    """
    tk = yf.Ticker('^TNX')
    hist = tk.history(period='1d')
    if hist.empty:
        return None
    return hist['Close'].iloc[-1] / 100


def main():
    ticker = input("Enter ticker (e.g. AAPL): ").upper().strip()
    # Get and display current US risk-free rate
    rfr = get_us_risk_free_rate()
    if rfr is not None:
        print(f"\nCurrent US risk-free rate (10Y Treasury): {rfr:.2%}\n")
    else:
        print("\nCould not retrieve US risk-free rate.\n")
    expirations = get_expirations(ticker)
    if not expirations:
        print(f"No expiration dates found for {ticker}")
        return

    print("\nAvailable expiration dates:")
    for idx, date in enumerate(expirations, start=1):
        print(f"  {idx}. {date}")

    try:
        choice = int(input("\nSelect expiration number: "))
        expiration = expirations[choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return

    print(f"\nFetching option chain for {ticker} - Expiration: {expiration}\n")
    calls, puts = get_option_chain(ticker, expiration)

    pd.set_option('display.max_rows', 10)
    print("== CALLS ==")
    print(calls)

    print("\n== PUTS ==")
    print(puts)


if __name__ == "__main__":
    main() 