from django.db import models
from django.core.validators import MinValueValidator
from typing import Literal
import numpy as np


class Option(models.Model):
    """
    Modelo Django para representar una opción financiera
    """
    TYPE_CHOICES = [
        ('call', 'Call'),
        ('put', 'Put'),
    ]
    
    STYLE_CHOICES = [
        ('european', 'European'),
        ('american', 'American'),
    ]
    
    # Campos básicos
    name = models.CharField(max_length=100, help_text="Nombre descriptivo de la opción")
    type = models.CharField(max_length=4, choices=TYPE_CHOICES, help_text="Tipo de opción")
    style = models.CharField(max_length=8, choices=STYLE_CHOICES, help_text="Estilo de ejercicio")
    
    # Parámetros de la opción
    spot = models.DecimalField(
        max_digits=15, 
        decimal_places=6, 
        validators=[MinValueValidator(0.000001)],
        help_text="Precio actual del activo subyacente"
    )
    strike = models.DecimalField(
        max_digits=15, 
        decimal_places=6, 
        validators=[MinValueValidator(0.000001)],
        help_text="Precio de ejercicio"
    )
    maturity = models.DecimalField(
        max_digits=8, 
        decimal_places=6, 
        validators=[MinValueValidator(0.000001)],
        help_text="Tiempo hasta vencimiento en años"
    )
    volatility = models.DecimalField(
        max_digits=8, 
        decimal_places=6, 
        validators=[MinValueValidator(0)],
        help_text="Volatilidad implícita"
    )
    rate = models.DecimalField(
        max_digits=8, 
        decimal_places=6, 
        validators=[MinValueValidator(0)],
        help_text="Tasa libre de riesgo"
    )
    
    # Campos adicionales
    market_price = models.DecimalField(
        max_digits=15, 
        decimal_places=6, 
        null=True, 
        blank=True,
        help_text="Precio de mercado observado"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Opción'
        verbose_name_plural = 'Opciones'
    
    def __str__(self):
        return f"{self.name} - {self.get_type_display()} {self.get_style_display()}"
    
    def payoff(self, S: float) -> float:
        """Calcula el payoff de la opción para un precio S"""
        if self.type == 'call':
            return max(S - float(self.strike), 0)
        else:
            return max(float(self.strike) - S, 0)
    
    @property
    def is_in_the_money(self) -> bool:
        """Determina si la opción está in-the-money"""
        if self.type == 'call':
            return float(self.spot) > float(self.strike)
        else:
            return float(self.spot) < float(self.strike)
    
    @property
    def moneyness(self) -> float:
        """Calcula el moneyness (S/K)"""
        return float(self.spot) / float(self.strike)


class Greeks(models.Model):
    """
    Modelo para almacenar las griegas calculadas de una opción
    """
    option = models.OneToOneField(Option, on_delete=models.CASCADE, related_name='greeks')
    
    # Griegas de primera orden
    delta = models.DecimalField(max_digits=15, decimal_places=8, help_text="Sensibilidad al precio del subyacente")
    
    # Griegas de segunda orden
    gamma = models.DecimalField(max_digits=15, decimal_places=8, help_text="Sensibilidad del delta")
    
    # Griegas de volatilidad
    vega = models.DecimalField(max_digits=15, decimal_places=8, help_text="Sensibilidad a la volatilidad")
    
    # Griegas de tiempo
    theta = models.DecimalField(max_digits=15, decimal_places=8, help_text="Decay temporal")
    
    # Griegas de tasa
    rho = models.DecimalField(max_digits=15, decimal_places=8, help_text="Sensibilidad a la tasa de interés")
    
    # Metadatos del cálculo
    pricing_model = models.CharField(max_length=20, help_text="Modelo utilizado para el cálculo")
    calculated_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Greeks'
        verbose_name_plural = 'Greeks'
    
    def __str__(self):
        return f"Greeks for {self.option.name}"
    
    def as_dict(self):
        """Retorna las griegas como diccionario"""
        return {
            'delta': float(self.delta),
            'gamma': float(self.gamma),
            'vega': float(self.vega),
            'theta': float(self.theta),
            'rho': float(self.rho)
        }


class Portfolio(models.Model):
    """
    Modelo para representar un portfolio de opciones
    """
    name = models.CharField(max_length=100, help_text="Nombre del portfolio")
    description = models.TextField(blank=True, help_text="Descripción del portfolio")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Portfolio'
        verbose_name_plural = 'Portfolios'
    
    def __str__(self):
        return self.name
    
    @property
    def total_positions(self):
        """Número total de posiciones en el portfolio"""
        return self.positions.count()
    
    def value(self, spot_prices: dict = None) -> float:
        """Calcula el valor total del portfolio"""
        total_value = 0
        for position in self.positions.all():
            if spot_prices and position.option.id in spot_prices:
                spot_price = spot_prices[position.option.id]
                payoff = position.option.payoff(spot_price)
            else:
                payoff = position.option.payoff(float(position.option.spot))
            total_value += payoff * float(position.quantity)
        return total_value


class PortfolioPosition(models.Model):
    """
    Modelo para representar una posición específica en un portfolio
    """
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='positions')
    option = models.ForeignKey(Option, on_delete=models.CASCADE)
    quantity = models.DecimalField(
        max_digits=15, 
        decimal_places=6,
        help_text="Cantidad de contratos (positivo = largo, negativo = corto)"
    )
    entry_price = models.DecimalField(
        max_digits=15, 
        decimal_places=6, 
        null=True, 
        blank=True,
        help_text="Precio de entrada de la posición"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['portfolio', 'option']
        verbose_name = 'Posición del Portfolio'
        verbose_name_plural = 'Posiciones del Portfolio'
    
    def __str__(self):
        return f"{self.portfolio.name} - {self.option.name} ({self.quantity})"
    
    @property
    def is_long(self) -> bool:
        """Determina si la posición es larga"""
        return float(self.quantity) > 0
    
    @property
    def is_short(self) -> bool:
        """Determina si la posición es corta"""
        return float(self.quantity) < 0
    
    @property
    def notional_value(self) -> float:
        """Calcula el valor nocional de la posición"""
        if self.entry_price:
            return float(self.quantity) * float(self.entry_price)
        return float(self.quantity) * float(self.option.market_price or 0)


class PricingResult(models.Model):
    """
    Modelo para almacenar resultados de pricing
    """
    PRICING_MODELS = [
        ('black_scholes', 'Black-Scholes'),
        ('binomial', 'Binomial'),
        ('monte_carlo', 'Monte Carlo'),
    ]
    
    option = models.ForeignKey(Option, on_delete=models.CASCADE, related_name='pricing_results')
    model = models.CharField(max_length=20, choices=PRICING_MODELS)
    calculated_price = models.DecimalField(max_digits=15, decimal_places=6)
    implied_volatility = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    
    # Parámetros específicos del modelo
    n_steps = models.IntegerField(null=True, blank=True, help_text="Número de pasos para modelos binomial/MC")
    n_simulations = models.IntegerField(null=True, blank=True, help_text="Número de simulaciones para MC")
    
    # Timestamp
    calculated_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-calculated_at']
        verbose_name = 'Resultado de Pricing'
        verbose_name_plural = 'Resultados de Pricing'
    
    def __str__(self):
        return f"{self.option.name} - {self.get_model_display()}: {self.calculated_price}"
