import numpy as np
from typing import Dict
from decimal import Decimal
from ..models import Option, Greeks as GreeksModel


class MonteCarloService:
    """
    Servicio para cálculos de pricing usando simulación Monte Carlo
    """
    
    @staticmethod
    def price(option: Option, n_sim: int = 10000, n_steps: int = 100, seed: int = None) -> float:
        """
        Calcula el precio de una opción usando Monte Carlo
        """
        if seed:
            np.random.seed(seed)
        
        S = float(option.spot)
        K = float(option.strike)
        T = float(option.maturity)
        r = float(option.rate)
        sigma = float(option.volatility)
        
        if T <= 0:
            return option.payoff(S)
        
        dt = T / n_steps
        
        if option.style == 'european':
            # Opción europea - solo simulamos hasta el vencimiento
            Z = np.random.standard_normal(n_sim)
            ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            
            if option.type == 'call':
                payoffs = np.maximum(ST - K, 0)
            else:  # put
                payoffs = np.maximum(K - ST, 0)
            
            # Descontar al valor presente
            price = np.exp(-r * T) * np.mean(payoffs)
            
        else:
            # Opción americana - método Longstaff-Schwartz simplificado
            prices = np.zeros((n_sim, n_steps + 1))
            prices[:, 0] = S
            
            # Simular paths
            for i in range(1, n_steps + 1):
                Z = np.random.standard_normal(n_sim)
                prices[:, i] = prices[:, i-1] * np.exp(
                    (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
                )
            
            # Calcular payoffs en cada paso
            if option.type == 'call':
                exercise_values = np.maximum(prices - K, 0)
            else:  # put
                exercise_values = np.maximum(K - prices, 0)
            
            # Backward induction simplificado
            cashflows = exercise_values[:, -1]  # Payoff en vencimiento
            
            for i in range(n_steps - 1, 0, -1):
                # Valor de continuar (descontado)
                continuation_value = np.exp(-r * dt) * cashflows
                
                # Valor de ejercer ahora
                exercise_value = exercise_values[:, i]
                
                # Decidir si ejercer (simplificado - ejercer si ITM)
                exercise_mask = exercise_value > continuation_value
                cashflows = np.where(exercise_mask, exercise_value, continuation_value)
            
            # Descontar al tiempo 0
            price = np.exp(-r * dt) * np.mean(cashflows)
        
        return max(price, 0)
    
    @staticmethod
    def greeks(option: Option, n_sim: int = 10000, n_steps: int = 100, 
              seed: int = None, h: float = None) -> Dict[str, float]:
        """
        Calcula las griegas usando diferencias finitas con Monte Carlo
        """
        if h is None:
            h = max(0.01 * float(option.spot), 0.01)
        
        # Precio original
        price = MonteCarloService.price(option, n_sim, n_steps, seed)
        
        # Delta
        option_up = Option(
            name=option.name,
            type=option.type,
            style=option.style,
            spot=option.spot + Decimal(h),
            strike=option.strike,
            maturity=option.maturity,
            volatility=option.volatility,
            rate=option.rate
        )
        option_down = Option(
            name=option.name,
            type=option.type,
            style=option.style,
            spot=option.spot - Decimal(h),
            strike=option.strike,
            maturity=option.maturity,
            volatility=option.volatility,
            rate=option.rate
        )
        
        price_up = MonteCarloService.price(option_up, n_sim, n_steps, seed)
        price_down = MonteCarloService.price(option_down, n_sim, n_steps, seed)
        delta = (price_up - price_down) / (2 * h)
        
        # Gamma
        gamma = (price_up - 2 * price + price_down) / (h ** 2)
        
        # Vega
        vega_bump = 0.01
        option_vega_up = Option(
            name=option.name,
            type=option.type,
            style=option.style,
            spot=option.spot,
            strike=option.strike,
            maturity=option.maturity,
            volatility=option.volatility + Decimal(vega_bump),
            rate=option.rate
        )
        option_vega_down = Option(
            name=option.name,
            type=option.type,
            style=option.style,
            spot=option.spot,
            strike=option.strike,
            maturity=option.maturity,
            volatility=option.volatility - Decimal(vega_bump),
            rate=option.rate
        )
        
        price_vega_up = MonteCarloService.price(option_vega_up, n_sim, n_steps, seed)
        price_vega_down = MonteCarloService.price(option_vega_down, n_sim, n_steps, seed)
        vega = (price_vega_up - price_vega_down) / (2 * vega_bump)
        
        # Theta
        theta_bump = 1.0 / 365.0  # 1 día
        if float(option.maturity) > theta_bump:
            option_theta = Option(
                name=option.name,
                type=option.type,
                style=option.style,
                spot=option.spot,
                strike=option.strike,
                maturity=option.maturity - Decimal(theta_bump),
                volatility=option.volatility,
                rate=option.rate
            )
            price_theta = MonteCarloService.price(option_theta, n_sim, n_steps, seed)
            theta = (price_theta - price) / theta_bump
        else:
            theta = 0.0
        
        # Rho
        rho_bump = 0.01
        option_rho_up = Option(
            name=option.name,
            type=option.type,
            style=option.style,
            spot=option.spot,
            strike=option.strike,
            maturity=option.maturity,
            volatility=option.volatility,
            rate=option.rate + Decimal(rho_bump)
        )
        option_rho_down = Option(
            name=option.name,
            type=option.type,
            style=option.style,
            spot=option.spot,
            strike=option.strike,
            maturity=option.maturity,
            volatility=option.volatility,
            rate=option.rate - Decimal(rho_bump)
        )
        
        price_rho_up = MonteCarloService.price(option_rho_up, n_sim, n_steps, seed)
        price_rho_down = MonteCarloService.price(option_rho_down, n_sim, n_steps, seed)
        rho = (price_rho_up - price_rho_down) / (2 * rho_bump)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,  # Vega por 1% de cambio en volatilidad
            'theta': theta,  # Theta por día
            'rho': rho / 100  # Rho por 1% de cambio en tasa
        }
    
    @staticmethod
    def save_greeks_to_db(option: Option, greeks_dict: Dict[str, float], 
                         n_sim: int, n_steps: int) -> GreeksModel:
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
                'pricing_model': f'monte_carlo_n{n_sim}_s{n_steps}'
            }
        )
        return greeks_obj
