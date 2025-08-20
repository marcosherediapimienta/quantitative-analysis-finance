from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from decimal import Decimal


class Stock(models.Model):
    """
    Modelo para representar una acción/stock
    """
    symbol = models.CharField(max_length=10, unique=True, help_text="Símbolo bursátil")
    name = models.CharField(max_length=200, help_text="Nombre completo de la empresa")
    sector = models.CharField(max_length=100, blank=True, help_text="Sector de la empresa")
    market_cap = models.BigIntegerField(null=True, blank=True, help_text="Capitalización de mercado")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['symbol']
        verbose_name = 'Stock'
        verbose_name_plural = 'Stocks'
    
    def __str__(self):
        return f"{self.symbol} - {self.name}"


class StockPrice(models.Model):
    """
    Modelo para precios históricos de acciones
    """
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='prices')
    date = models.DateField()
    open_price = models.DecimalField(max_digits=15, decimal_places=6)
    high_price = models.DecimalField(max_digits=15, decimal_places=6)
    low_price = models.DecimalField(max_digits=15, decimal_places=6)
    close_price = models.DecimalField(max_digits=15, decimal_places=6)
    adj_close_price = models.DecimalField(max_digits=15, decimal_places=6)
    volume = models.BigIntegerField()
    
    class Meta:
        unique_together = ['stock', 'date']
        ordering = ['-date']
        verbose_name = 'Stock Price'
        verbose_name_plural = 'Stock Prices'
    
    def __str__(self):
        return f"{self.stock.symbol} - {self.date}: {self.close_price}"


class TechnicalIndicator(models.Model):
    """
    Modelo para indicadores técnicos calculados
    """
    INDICATOR_CHOICES = [
        ('sma', 'Simple Moving Average'),
        ('ema', 'Exponential Moving Average'),
        ('rsi', 'Relative Strength Index'),
        ('macd', 'MACD'),
        ('bollinger_upper', 'Bollinger Band Upper'),
        ('bollinger_lower', 'Bollinger Band Lower'),
        ('adx', 'Average Directional Index'),
        ('stochastic_k', 'Stochastic %K'),
        ('stochastic_d', 'Stochastic %D'),
    ]
    
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='indicators')
    date = models.DateField()
    indicator_type = models.CharField(max_length=20, choices=INDICATOR_CHOICES)
    period = models.IntegerField(help_text="Período usado para el cálculo")
    value = models.DecimalField(max_digits=15, decimal_places=6)
    
    # Timestamp
    calculated_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['stock', 'date', 'indicator_type', 'period']
        ordering = ['-date']
        verbose_name = 'Technical Indicator'
        verbose_name_plural = 'Technical Indicators'
    
    def __str__(self):
        return f"{self.stock.symbol} - {self.get_indicator_type_display()} ({self.period}): {self.value}"


class FinancialStatement(models.Model):
    """
    Modelo base para estados financieros
    """
    STATEMENT_TYPES = [
        ('income_statement', 'Income Statement'),
        ('balance_sheet', 'Balance Sheet'),
        ('cash_flow', 'Cash Flow Statement'),
    ]
    
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='financial_statements')
    statement_type = models.CharField(max_length=20, choices=STATEMENT_TYPES)
    period_end = models.DateField(help_text="Fecha de fin del período")
    fiscal_year = models.IntegerField()
    fiscal_quarter = models.IntegerField(null=True, blank=True, validators=[MinValueValidator(1), MaxValueValidator(4)])
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['stock', 'statement_type', 'period_end']
        ordering = ['-period_end']
        verbose_name = 'Financial Statement'
        verbose_name_plural = 'Financial Statements'
    
    def __str__(self):
        return f"{self.stock.symbol} - {self.get_statement_type_display()} - {self.period_end}"


class FinancialMetric(models.Model):
    """
    Modelo para métricas financieras específicas
    """
    financial_statement = models.ForeignKey(FinancialStatement, on_delete=models.CASCADE, related_name='metrics')
    metric_name = models.CharField(max_length=100, help_text="Nombre de la métrica financiera")
    value = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    
    class Meta:
        unique_together = ['financial_statement', 'metric_name']
        verbose_name = 'Financial Metric'
        verbose_name_plural = 'Financial Metrics'
    
    def __str__(self):
        return f"{self.financial_statement.stock.symbol} - {self.metric_name}: {self.value}"


class RiskMetric(models.Model):
    """
    Modelo para métricas de riesgo calculadas
    """
    METRIC_TYPES = [
        ('var', 'Value at Risk'),
        ('cvar', 'Conditional Value at Risk'),
        ('beta', 'Beta'),
        ('alpha', 'Alpha'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('sortino_ratio', 'Sortino Ratio'),
        ('max_drawdown', 'Maximum Drawdown'),
        ('volatility', 'Volatility'),
    ]
    
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='risk_metrics')
    metric_type = models.CharField(max_length=20, choices=METRIC_TYPES)
    value = models.DecimalField(max_digits=15, decimal_places=6)
    confidence_level = models.DecimalField(
        max_digits=5, 
        decimal_places=4, 
        null=True, 
        blank=True,
        help_text="Nivel de confianza para VaR/CVaR"
    )
    time_horizon = models.IntegerField(
        null=True, 
        blank=True,
        help_text="Horizonte temporal en días"
    )
    calculation_date = models.DateField()
    
    # Timestamp
    calculated_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['stock', 'metric_type', 'calculation_date', 'confidence_level', 'time_horizon']
        ordering = ['-calculation_date']
        verbose_name = 'Risk Metric'
        verbose_name_plural = 'Risk Metrics'
    
    def __str__(self):
        return f"{self.stock.symbol} - {self.get_metric_type_display()}: {self.value}"


class PortfolioOptimization(models.Model):
    """
    Modelo para resultados de optimización de portfolios
    """
    OPTIMIZATION_TYPES = [
        ('markowitz', 'Markowitz Mean-Variance'),
        ('black_litterman', 'Black-Litterman'),
        ('risk_parity', 'Risk Parity'),
        ('minimum_variance', 'Minimum Variance'),
        ('maximum_sharpe', 'Maximum Sharpe Ratio'),
    ]
    
    name = models.CharField(max_length=100, help_text="Nombre del portfolio optimizado")
    optimization_type = models.CharField(max_length=20, choices=OPTIMIZATION_TYPES)
    stocks = models.ManyToManyField(Stock, through='PortfolioWeight')
    
    # Parámetros de optimización
    expected_return = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    volatility = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    sharpe_ratio = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    
    # Metadatos
    optimization_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-optimization_date']
        verbose_name = 'Portfolio Optimization'
        verbose_name_plural = 'Portfolio Optimizations'
    
    def __str__(self):
        return f"{self.name} - {self.get_optimization_type_display()}"


class PortfolioWeight(models.Model):
    """
    Modelo para pesos de stocks en portfolios optimizados
    """
    portfolio = models.ForeignKey(PortfolioOptimization, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    weight = models.DecimalField(
        max_digits=8, 
        decimal_places=6,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Peso del stock en el portfolio (0-1)"
    )
    
    class Meta:
        unique_together = ['portfolio', 'stock']
        verbose_name = 'Portfolio Weight'
        verbose_name_plural = 'Portfolio Weights'
    
    def __str__(self):
        return f"{self.portfolio.name} - {self.stock.symbol}: {self.weight:.2%}"


class MarketData(models.Model):
    """
    Modelo para datos de mercado generales
    """
    date = models.DateField(unique=True)
    risk_free_rate = models.DecimalField(max_digits=8, decimal_places=6, help_text="Tasa libre de riesgo")
    market_return = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    vix = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True, help_text="Índice de volatilidad VIX")
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-date']
        verbose_name = 'Market Data'
        verbose_name_plural = 'Market Data'
    
    def __str__(self):
        return f"Market Data - {self.date}"
