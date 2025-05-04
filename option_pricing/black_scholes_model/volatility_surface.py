import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from black_scholes_model.implied_volatility import implied_volatility_newton, black_scholes_call_price

def get_option_chain(ticker: str, expiration: str) -> tuple:
    """
    Get option chain data for a specific expiration date.
    
    Args:
        ticker: Stock ticker symbol
        expiration: Option expiration date (YYYY-MM-DD)
    
    Returns:
        tuple: (calls DataFrame, puts DataFrame)
    """
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiration)
    return chain.calls, chain.puts

def calculate_volatility_surface(ticker: str, expirations: list = None) -> pd.DataFrame:
    """
    Calculate implied volatility surface for different strikes and expirations.
    
    Args:
        ticker: Stock ticker symbol
        expirations: List of expiration dates to analyze
    
    Returns:
        DataFrame: Volatility surface data
    """
    stock = yf.Ticker(ticker)
    current_price = stock.info['regularMarketPrice']
    
    # Get available expirations if not provided
    if expirations is None:
        expirations = stock.options[:4]  # Use first 4 expirations by default
    
    # Initialize lists to store data
    data = []
    
    for exp in expirations:
        # Get option chain
        calls, puts = get_option_chain(ticker, exp)
        
        # Calculate time to expiration in years
        exp_date = pd.to_datetime(exp)
        today = pd.Timestamp.today()
        T = (exp_date - today).days / 365
        
        # Get risk-free rate (using 10-year Treasury yield)
        r = yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1] / 100
        
        # Filter options within reasonable moneyness range
        calls = calls[(calls['strike'] >= current_price * 0.7) & 
                     (calls['strike'] <= current_price * 1.3)]
        
        # Calculate implied volatility for each strike
        for _, row in calls.iterrows():
            try:
                iv = implied_volatility_newton(
                    market_price=row['lastPrice'],
                    S=current_price,
                    K=row['strike'],
                    T=T,
                    r=r
                )
                if iv is not None:
                    data.append({
                        'Expiration': exp,
                        'Strike': row['strike'],
                        'IV': iv,
                        'Moneyness': row['strike'] / current_price
                    })
            except:
                continue
    
    return pd.DataFrame(data)

def plot_volatility_surface(df: pd.DataFrame, save_path='volatility_surface.png'):
    """
    Create 3D surface plot of implied volatility and save it to a file.
    
    Args:
        df: DataFrame containing volatility surface data
        save_path: Path where to save the plot
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert expiration dates to numeric values for plotting
    exp_dates = pd.to_datetime(df['Expiration'].unique())
    exp_nums = np.arange(len(exp_dates))
    exp_map = dict(zip(exp_dates, exp_nums))
    df['ExpirationNum'] = df['Expiration'].map(exp_map)
    
    # Create surface plot
    x = df['Moneyness']
    y = df['ExpirationNum']
    z = df['IV']
    
    # Create triangulation
    triang = tri.Triangulation(x, y)
    
    # Plot surface
    surf = ax.plot_trisurf(x, y, z, triangles=triang.triangles, cmap='viridis')
    
    # Customize plot
    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('Time to Expiration')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Volatility Surface')
    
    # Set y-axis ticks to show actual dates
    ax.set_yticks(exp_nums)
    ax.set_yticklabels([d.strftime('%Y-%m-%d') for d in exp_dates])
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    print(f"Volatility surface plot saved to {save_path}")

def plot_volatility_smile(df: pd.DataFrame, expiration: str = None, save_path='volatility_smile.png'):
    """
    Create 2D plot of volatility smile and save it to a file.
    
    Args:
        df: DataFrame containing volatility surface data
        expiration: Specific expiration date to plot (if None, uses first available)
        save_path: Path where to save the plot
    """
    if expiration is None:
        expiration = df['Expiration'].iloc[0]
    
    # Filter data for specific expiration
    exp_data = df[df['Expiration'] == expiration]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(exp_data['Moneyness'], exp_data['IV'], 'b-', label='Volatility Smile')
    plt.scatter(exp_data['Moneyness'], exp_data['IV'], c='blue', alpha=0.5)
    
    # Add vertical line at moneyness = 1 (at-the-money)
    plt.axvline(x=1, color='r', linestyle='--', label='At-the-money')
    
    # Customize plot
    plt.xlabel('Moneyness (K/S)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Volatility Smile - {expiration}')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    print(f"Volatility smile plot saved to {save_path}")

def print_summary(df: pd.DataFrame, ticker: str):
    """
    Print and save a summary of the volatility analysis.
    
    Args:
        df: DataFrame containing volatility surface data
        ticker: Stock ticker symbol
    """
    # Get current stock price
    stock = yf.Ticker(ticker)
    current_price = stock.info['regularMarketPrice']
    
    # Create summary
    summary = []
    summary.append(f"Volatility Analysis Summary for {ticker}")
    summary.append(f"Current Stock Price: ${current_price:.2f}")
    summary.append("\nExpirations Used:")
    
    for exp in sorted(df['Expiration'].unique()):
        exp_data = df[df['Expiration'] == exp]
        days_to_exp = (pd.to_datetime(exp) - pd.Timestamp.today()).days
        summary.append(f"\n{exp} (Days to Expiration: {days_to_exp})")
        summary.append("Strikes analyzed:")
        strikes = sorted(exp_data['Strike'].unique())
        summary.append("  ITM Strikes: " + ", ".join([f"${k:.1f}" for k in strikes if k < current_price]))
        summary.append("  ATM Strikes: " + ", ".join([f"${k:.1f}" for k in strikes if k == current_price]))
        summary.append("  OTM Strikes: " + ", ".join([f"${k:.1f}" for k in strikes if k > current_price]))
        summary.append(f"  Total strikes: {len(strikes)}")
        
        # Add IV range for this expiration
        iv_min = exp_data['IV'].min()
        iv_max = exp_data['IV'].max()
        summary.append(f"  IV Range: {iv_min:.1%} - {iv_max:.1%}")
    
    # Print summary
    print("\n".join(summary))
    
    # Save summary to file
    with open('volatility_analysis_summary.txt', 'w') as f:
        f.write("\n".join(summary))
    print("\nSummary saved to volatility_analysis_summary.txt")

def main():
    # Example usage
    ticker = "NVDA"
    
    # Calculate volatility surface
    print(f"Calculating volatility surface for {ticker}...")
    vol_surface = calculate_volatility_surface(ticker)
    
    # Print and save summary
    print("\nAnalysis Summary:")
    print_summary(vol_surface, ticker)
    
    # Save results to CSV
    vol_surface.to_csv('volatility_data.csv', index=False)
    print("\nVolatility data saved to volatility_data.csv")
    
    # Plot and save results
    print("\nCreating volatility surface plot...")
    plot_volatility_surface(vol_surface)
    
    print("\nCreating volatility smile plot...")
    plot_volatility_smile(vol_surface)

if __name__ == "__main__":
    main() 