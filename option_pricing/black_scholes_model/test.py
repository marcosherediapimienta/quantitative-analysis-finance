import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

class OptionsAcademicAnalysis:
    def __init__(self, S: float, r: float, sigma: float):
        """
        Initialize the options analysis class with custom parameters.
        
        Args:
            S: Current stock price
            r: Risk-free rate (in decimal)
            sigma: Volatility (in decimal)
        """
        self.S = S
        self.r = r
        self.sigma = sigma
        
    def calculate_greeks(self, option_type: str, K: float, T: float) -> dict:
        """
        Calculate option Greeks.
        
        Args:
            option_type: 'call' or 'put'
            K: Strike price
            T: Time to expiration (in years)
        
        Returns:
            dict: Option Greeks
        """
        d1 = (np.log(self.S/K) + (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = self.S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(T))
            theta = (-self.S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
            vega = self.S * np.sqrt(T) * norm.pdf(d1)
            rho = K * T * np.exp(-self.r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(T))
            theta = (-self.S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
            vega = self.S * np.sqrt(T) * norm.pdf(d1)
            rho = -K * T * np.exp(-self.r * T) * norm.cdf(-d2)
        
        return {
            'Price': price,
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }
    
    def analyze_scenario(self, option_type: str, K: float, T: float) -> pd.DataFrame:
        """
        Analyze a specific option scenario.
        
        Args:
            option_type: 'call' or 'put'
            K: Strike price
            T: Time to expiration (in years)
        
        Returns:
            DataFrame: Scenario analysis
        """
        greeks = self.calculate_greeks(option_type, K, T)
        
        # Create scenario analysis
        analysis = pd.DataFrame({
            'Parameter': ['Stock Price', 'Strike Price', 'Time to Exp', 'Risk-free Rate', 'Volatility',
                         'Option Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
            'Value': [self.S, K, T, self.r, self.sigma,
                     greeks['Price'], greeks['Delta'], greeks['Gamma'], 
                     greeks['Theta'], greeks['Vega'], greeks['Rho']]
        })
        
        # Format values
        analysis['Value'] = analysis['Value'].apply(lambda x: f"${x:.2f}" if x > 10 else f"{x:.4f}")
        
        return analysis
    
    def plot_price_sensitivity(self, option_type: str, K: float, T: float, 
                             save_path: str = 'price_sensitivity.png'):
        """
        Plot option price sensitivity to different parameters.
        
        Args:
            option_type: 'call' or 'put'
            K: Strike price
            T: Time to expiration (in years)
            save_path: Path to save the plot
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Price vs Stock Price
        S_range = np.linspace(self.S * 0.5, self.S * 1.5, 100)
        prices = [self.calculate_greeks(option_type, K, T)['Price'] for S in S_range]
        ax1.plot(S_range, prices)
        ax1.set_title('Price vs Stock Price')
        ax1.set_xlabel('Stock Price')
        ax1.set_ylabel('Option Price')
        ax1.grid(True)
        
        # 2. Price vs Time to Expiration
        T_range = np.linspace(0.01, T * 2, 100)
        prices = [self.calculate_greeks(option_type, K, t)['Price'] for t in T_range]
        ax2.plot(T_range, prices)
        ax2.set_title('Price vs Time to Expiration')
        ax2.set_xlabel('Time to Expiration (years)')
        ax2.set_ylabel('Option Price')
        ax2.grid(True)
        
        # 3. Price vs Volatility
        sigma_range = np.linspace(0.1, self.sigma * 2, 100)
        prices = [self.calculate_greeks(option_type, K, T)['Price'] for sigma in sigma_range]
        ax3.plot(sigma_range, prices)
        ax3.set_title('Price vs Volatility')
        ax3.set_xlabel('Volatility')
        ax3.set_ylabel('Option Price')
        ax3.grid(True)
        
        # 4. Price vs Strike Price
        K_range = np.linspace(K * 0.5, K * 1.5, 100)
        prices = [self.calculate_greeks(option_type, k, T)['Price'] for k in K_range]
        ax4.plot(K_range, prices)
        ax4.set_title('Price vs Strike Price')
        ax4.set_xlabel('Strike Price')
        ax4.set_ylabel('Option Price')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_greeks_sensitivity(self, option_type: str, K: float, T: float,
                              save_path: str = 'greeks_sensitivity.png'):
        """
        Plot Greeks sensitivity to different parameters.
        
        Args:
            option_type: 'call' or 'put'
            K: Strike price
            T: Time to expiration (in years)
            save_path: Path to save the plot
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Delta vs Stock Price
        S_range = np.linspace(self.S * 0.5, self.S * 1.5, 100)
        deltas = [self.calculate_greeks(option_type, K, T)['Delta'] for S in S_range]
        ax1.plot(S_range, deltas)
        ax1.set_title('Delta vs Stock Price')
        ax1.set_xlabel('Stock Price')
        ax1.set_ylabel('Delta')
        ax1.grid(True)
        
        # 2. Gamma vs Stock Price
        gammas = [self.calculate_greeks(option_type, K, T)['Gamma'] for S in S_range]
        ax2.plot(S_range, gammas)
        ax2.set_title('Gamma vs Stock Price')
        ax2.set_xlabel('Stock Price')
        ax2.set_ylabel('Gamma')
        ax2.grid(True)
        
        # 3. Theta vs Time to Expiration
        T_range = np.linspace(0.01, T * 2, 100)
        thetas = [self.calculate_greeks(option_type, K, t)['Theta'] for t in T_range]
        ax3.plot(T_range, thetas)
        ax3.set_title('Theta vs Time to Expiration')
        ax3.set_xlabel('Time to Expiration (years)')
        ax3.set_ylabel('Theta')
        ax3.grid(True)
        
        # 4. Vega vs Volatility
        sigma_range = np.linspace(0.1, self.sigma * 2, 100)
        vegas = [self.calculate_greeks(option_type, K, T)['Vega'] for sigma in sigma_range]
        ax4.plot(sigma_range, vegas)
        ax4.set_title('Vega vs Volatility')
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Vega')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    # Example usage with custom parameters
    S = 105.0  # Stock price
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    analysis = OptionsAcademicAnalysis(S, r, sigma)
    
    # Analyze a specific scenario
    option_type = 'call'
    K = 100.0  # Strike price
    T = 30/365    # Time to expiration (1 year)
    
    # Print scenario analysis
    print("\nScenario Analysis:")
    print(analysis.analyze_scenario(option_type, K, T))
    
    # Generate sensitivity plots
    analysis.plot_price_sensitivity(option_type, K, T)
    analysis.plot_greeks_sensitivity(option_type, K, T)

if __name__ == "__main__":
    main() 