import numpy as np
from typing import Dict
from decimal import Decimal
from ..models import Option, Greeks as GreeksModel


class BinomialService:
    """
    Servicio para cálculos de pricing usando el modelo Binomial
    """
    
    @staticmethod
    def price(option: Option, N: int = 1000) -> float:
        """
        Calcula el precio de una opción usando el modelo Binomial
        """
        S = float(option.spot)
        K = float(option.strike)
        T = float(option.maturity)
        r = float(option.rate)
        sigma = float(option.volatility)
        
        if T <= 0:
            return option.payoff(S)
        
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        # Precios del activo subyacente en el vencimiento
        j = np.arange(N + 1)
        ST = S * (u ** j) * (d ** (N - j))
        
        # Payoffs en el vencimiento
        if option.type == 'call':
            payoff = np.maximum(ST - K, 0)
        else:  # put
            payoff = np.maximum(K - ST, 0)
        
        # Backward induction
        if option.style == 'european':
            # Opción europea - solo ejercicio en vencimiento
            for step in range(N):
                payoff = discount * (p * payoff[1:] + (1 - p) * payoff[:-1])
        else:
            # Opción americana - posible ejercicio early
            for step in range(N):
                # Valor de continuación
                payoff = discount * (p * payoff[1:] + (1 - p) * payoff[:-1])
                
                # Precios del subyacente en este paso
                j = np.arange(N - step)
                S_step = S * (u ** j) * (d ** (N - step - 1 - j))
                
                # Valor de ejercicio early
                if option.type == 'call':
                    exercise_value = np.maximum(S_step - K, 0)
                else:  # put
                    exercise_value = np.maximum(K - S_step, 0)
                
                # Tomar el máximo entre continuar y ejercer
                payoff = np.maximum(payoff, exercise_value)
        
        return float(payoff[0])
    
    @staticmethod
    def greeks(option: Option, N: int = 1000, h: float = None) -> Dict[str, float]:
        """
        Calcula las griegas usando diferencias finitas en el modelo Binomial
        """
        if h is None:
            h = max(0.01 * float(option.spot), 0.01)
        
        # Precio original
        price = BinomialService.price(option, N)
        
        # Delta - sensibilidad al precio del subyacente
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
        
        price_up = BinomialService.price(option_up, N)
        price_down = BinomialService.price(option_down, N)
        delta = (price_up - price_down) / (2 * h)
        
        # Gamma - segunda derivada respecto al precio
        gamma = (price_up - 2 * price + price_down) / (h ** 2)
        
        # Vega - sensibilidad a la volatilidad
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
        
        price_vega_up = BinomialService.price(option_vega_up, N)
        price_vega_down = BinomialService.price(option_vega_down, N)
        vega = (price_vega_up - price_vega_down) / (2 * vega_bump)
        
        # Theta - sensibilidad al tiempo
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
            price_theta = BinomialService.price(option_theta, N)
            theta = (price_theta - price) / theta_bump
        else:
            theta = 0.0
        
        # Rho - sensibilidad a la tasa de interés
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
        
        price_rho_up = BinomialService.price(option_rho_up, N)
        price_rho_down = BinomialService.price(option_rho_down, N)
        rho = (price_rho_up - price_rho_down) / (2 * rho_bump)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,  # Vega por 1% de cambio en volatilidad
            'theta': theta,  # Theta por día
            'rho': rho / 100  # Rho por 1% de cambio en tasa
        }
    
    @staticmethod
    def save_greeks_to_db(option: Option, greeks_dict: Dict[str, float], N: int) -> GreeksModel:
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
                'pricing_model': f'binomial_N{N}'
            }
        )
        return greeks_obj
