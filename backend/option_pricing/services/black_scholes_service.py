import numpy as np
from scipy.stats import norm
from typing import Dict, Any
from decimal import Decimal
from ..models import Option, Greeks as GreeksModel


class BlackScholesService:
    """
    Servicio para cálculos de pricing usando el modelo Black-Scholes
    """
    
    @staticmethod
    def price(option: Option) -> float:
        """
        Calcula el precio de una opción usando Black-Scholes
        """
        S = float(option.spot)
        K = float(option.strike)
        T = float(option.maturity)
        r = float(option.rate)
        sigma = float(option.volatility)
        
        if T <= 0:
            # Opción vencida
            return option.payoff(S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option.type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)  # El precio no puede ser negativo
    
    @staticmethod
    def greeks(option: Option) -> Dict[str, float]:
        """
        Calcula las griegas usando Black-Scholes
        """
        S = float(option.spot)
        K = float(option.strike)
        T = float(option.maturity)
        r = float(option.rate)
        sigma = float(option.volatility)
        
        if T <= 0:
            # Opción vencida - griegas son cero excepto delta
            delta = 1.0 if (option.type == 'call' and S > K) or (option.type == 'put' and S < K) else 0.0
            return {
                'delta': delta,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option.type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma (igual para calls y puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (igual para calls y puts)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        # Theta
        if option.type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:  # put
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        # Rho
        if option.type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,  # Vega por 1% de cambio en volatilidad
            'theta': theta / 365,  # Theta por día
            'rho': rho / 100  # Rho por 1% de cambio en tasa
        }
    
    @staticmethod
    def implied_volatility(option: Option, market_price: float, max_iterations: int = 100, 
                          tolerance: float = 1e-6) -> float:
        """
        Calcula la volatilidad implícita usando el método de Newton-Raphson
        """
        if market_price <= 0:
            return 0.0
        
        # Estimación inicial
        sigma = 0.2
        
        for i in range(max_iterations):
            # Crear opción temporal con volatilidad actual
            temp_option = Option(
                name=option.name,
                type=option.type,
                style=option.style,
                spot=option.spot,
                strike=option.strike,
                maturity=option.maturity,
                volatility=Decimal(sigma),
                rate=option.rate
            )
            
            # Calcular precio y vega
            price = BlackScholesService.price(temp_option)
            greeks = BlackScholesService.greeks(temp_option)
            vega = greeks['vega'] * 100  # Vega real (no escalado)
            
            # Diferencia de precio
            price_diff = price - market_price
            
            # Convergencia
            if abs(price_diff) < tolerance:
                return sigma
            
            # Newton-Raphson step
            if vega != 0:
                sigma = sigma - price_diff / vega
            else:
                break
            
            # Mantener sigma en rango válido
            sigma = max(0.001, min(sigma, 5.0))
        
        return sigma
    
    @staticmethod
    def save_greeks_to_db(option: Option, greeks_dict: Dict[str, float]) -> GreeksModel:
        """
        Guarda las griegas calculadas en la base de datos
        """
        greeks_obj, created = GreeksModel.objects.update_or_create(
            option=option,
            defaults={
                'delta': Decimal(str(greeks_dict['delta'])),
                'gamma': Decimal(str(greeks_dict['gamma'])),
                'vega': Decimal(str(greeks_dict['vega'])),
                'theta': Decimal(str(greeks_dict['theta'])),
                'rho': Decimal(str(greeks_dict['rho'])),
                'pricing_model': 'black_scholes'
            }
        )
        return greeks_obj
