from rest_framework import serializers
from decimal import Decimal
from .models import Option, Greeks, Portfolio, PortfolioPosition, PricingResult


class GreeksSerializer(serializers.ModelSerializer):
    """
    Serializer para las griegas de una opción
    """
    class Meta:
        model = Greeks
        fields = ['delta', 'gamma', 'vega', 'theta', 'rho', 'pricing_model', 'calculated_at']
        read_only_fields = ['calculated_at']


class OptionSerializer(serializers.ModelSerializer):
    """
    Serializer para opciones financieras
    """
    greeks = GreeksSerializer(read_only=True)
    is_in_the_money = serializers.ReadOnlyField()
    moneyness = serializers.ReadOnlyField()
    
    class Meta:
        model = Option
        fields = [
            'id', 'name', 'type', 'style', 'spot', 'strike', 'maturity',
            'volatility', 'rate', 'market_price', 'created_at', 'updated_at',
            'greeks', 'is_in_the_money', 'moneyness'
        ]
        read_only_fields = ['created_at', 'updated_at']
    
    def validate(self, data):
        """
        Validaciones personalizadas
        """
        if data.get('spot', 0) <= 0:
            raise serializers.ValidationError("El precio spot debe ser positivo")
        
        if data.get('strike', 0) <= 0:
            raise serializers.ValidationError("El precio strike debe ser positivo")
        
        if data.get('maturity', 0) <= 0:
            raise serializers.ValidationError("El tiempo a vencimiento debe ser positivo")
        
        if data.get('volatility', 0) < 0:
            raise serializers.ValidationError("La volatilidad no puede ser negativa")
        
        if data.get('rate', 0) < 0:
            raise serializers.ValidationError("La tasa libre de riesgo no puede ser negativa")
        
        return data


class OptionCreateSerializer(serializers.ModelSerializer):
    """
    Serializer específico para crear opciones
    """
    class Meta:
        model = Option
        fields = [
            'name', 'type', 'style', 'spot', 'strike', 'maturity',
            'volatility', 'rate', 'market_price'
        ]


class PricingRequestSerializer(serializers.Serializer):
    """
    Serializer para requests de pricing
    """
    PRICING_MODELS = [
        ('black_scholes', 'Black-Scholes'),
        ('binomial', 'Binomial'),
        ('monte_carlo', 'Monte Carlo'),
    ]
    
    model = serializers.ChoiceField(choices=PRICING_MODELS, default='black_scholes')
    calculate_greeks = serializers.BooleanField(default=True)
    
    # Parámetros específicos del modelo
    n_steps = serializers.IntegerField(default=100, min_value=10, max_value=10000)
    n_simulations = serializers.IntegerField(default=10000, min_value=1000, max_value=100000)
    seed = serializers.IntegerField(required=False, allow_null=True)


class PricingResponseSerializer(serializers.Serializer):
    """
    Serializer para respuestas de pricing
    """
    option_id = serializers.IntegerField()
    model = serializers.CharField()
    calculated_price = serializers.DecimalField(max_digits=15, decimal_places=6)
    implied_volatility = serializers.DecimalField(max_digits=8, decimal_places=6, allow_null=True)
    greeks = GreeksSerializer(allow_null=True)
    calculation_time = serializers.FloatField()


class PortfolioPositionSerializer(serializers.ModelSerializer):
    """
    Serializer para posiciones del portfolio
    """
    option = OptionSerializer(read_only=True)
    option_id = serializers.IntegerField(write_only=True)
    is_long = serializers.ReadOnlyField()
    is_short = serializers.ReadOnlyField()
    notional_value = serializers.ReadOnlyField()
    
    class Meta:
        model = PortfolioPosition
        fields = [
            'id', 'option', 'option_id', 'quantity', 'entry_price',
            'is_long', 'is_short', 'notional_value', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class PortfolioSerializer(serializers.ModelSerializer):
    """
    Serializer para portfolios
    """
    positions = PortfolioPositionSerializer(many=True, read_only=True)
    total_positions = serializers.ReadOnlyField()
    
    class Meta:
        model = Portfolio
        fields = [
            'id', 'name', 'description', 'positions', 'total_positions',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class PortfolioCreateSerializer(serializers.ModelSerializer):
    """
    Serializer para crear portfolios
    """
    positions = PortfolioPositionSerializer(many=True, required=False)
    
    class Meta:
        model = Portfolio
        fields = ['name', 'description', 'positions']
    
    def create(self, validated_data):
        positions_data = validated_data.pop('positions', [])
        portfolio = Portfolio.objects.create(**validated_data)
        
        for position_data in positions_data:
            PortfolioPosition.objects.create(portfolio=portfolio, **position_data)
        
        return portfolio


class PortfolioAnalysisRequestSerializer(serializers.Serializer):
    """
    Serializer para requests de análisis de portfolio
    """
    PRICING_MODELS = [
        ('black_scholes', 'Black-Scholes'),
        ('binomial', 'Binomial'),
        ('monte_carlo', 'Monte Carlo'),
    ]
    
    model = serializers.ChoiceField(choices=PRICING_MODELS, default='black_scholes')
    calculate_var = serializers.BooleanField(default=True)
    confidence_level = serializers.FloatField(default=0.99, min_value=0.90, max_value=0.999)
    horizon_days = serializers.IntegerField(default=1, min_value=1, max_value=252)
    n_simulations = serializers.IntegerField(default=10000, min_value=1000, max_value=100000)
    n_steps = serializers.IntegerField(default=100, min_value=10, max_value=1000)


class PortfolioAnalysisResponseSerializer(serializers.Serializer):
    """
    Serializer para respuestas de análisis de portfolio
    """
    portfolio_id = serializers.IntegerField()
    total_value = serializers.DecimalField(max_digits=20, decimal_places=6)
    total_greeks = GreeksSerializer()
    var = serializers.DecimalField(max_digits=20, decimal_places=6, allow_null=True)
    expected_shortfall = serializers.DecimalField(max_digits=20, decimal_places=6, allow_null=True)
    model_used = serializers.CharField()
    calculation_time = serializers.FloatField()


class PricingResultSerializer(serializers.ModelSerializer):
    """
    Serializer para resultados de pricing almacenados
    """
    option = OptionSerializer(read_only=True)
    
    class Meta:
        model = PricingResult
        fields = [
            'id', 'option', 'model', 'calculated_price', 'implied_volatility',
            'n_steps', 'n_simulations', 'calculated_at'
        ]
        read_only_fields = ['calculated_at']


class ImpliedVolatilityRequestSerializer(serializers.Serializer):
    """
    Serializer para requests de volatilidad implícita
    """
    market_price = serializers.DecimalField(max_digits=15, decimal_places=6, min_value=Decimal('0.01'))
    max_iterations = serializers.IntegerField(default=100, min_value=10, max_value=1000)
    tolerance = serializers.FloatField(default=1e-6, min_value=1e-10, max_value=1e-3)


class ImpliedVolatilityResponseSerializer(serializers.Serializer):
    """
    Serializer para respuestas de volatilidad implícita
    """
    option_id = serializers.IntegerField()
    market_price = serializers.DecimalField(max_digits=15, decimal_places=6)
    implied_volatility = serializers.DecimalField(max_digits=8, decimal_places=6)
    iterations_used = serializers.IntegerField()
    converged = serializers.BooleanField()


class SensitivityAnalysisRequestSerializer(serializers.Serializer):
    """
    Serializer para requests de análisis de sensibilidad
    """
    SENSITIVITY_TYPES = [
        ('spot', 'Precio del Subyacente'),
        ('strike', 'Precio de Ejercicio'),
        ('volatility', 'Volatilidad'),
        ('rate', 'Tasa de Interés'),
        ('time', 'Tiempo al Vencimiento'),
    ]
    
    sensitivity_type = serializers.ChoiceField(choices=SENSITIVITY_TYPES)
    min_value = serializers.FloatField()
    max_value = serializers.FloatField()
    num_points = serializers.IntegerField(default=21, min_value=5, max_value=101)
    model = serializers.ChoiceField(
        choices=PricingRequestSerializer.PRICING_MODELS, 
        default='black_scholes'
    )
    
    def validate(self, data):
        if data['min_value'] >= data['max_value']:
            raise serializers.ValidationError("min_value debe ser menor que max_value")
        return data


class SensitivityAnalysisResponseSerializer(serializers.Serializer):
    """
    Serializer para respuestas de análisis de sensibilidad
    """
    option_id = serializers.IntegerField()
    sensitivity_type = serializers.CharField()
    model_used = serializers.CharField()
    data_points = serializers.ListField(
        child=serializers.DictField()
    )
    calculation_time = serializers.FloatField()
