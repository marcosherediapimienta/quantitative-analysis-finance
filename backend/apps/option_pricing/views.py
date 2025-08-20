import time
from decimal import Decimal
from typing import Dict, Any

from django.core.cache import cache
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Option, Portfolio, PortfolioPosition, Greeks, PricingResult
from .serializers import (
    OptionSerializer, OptionCreateSerializer, PortfolioSerializer, 
    PortfolioCreateSerializer, PortfolioPositionSerializer,
    PricingRequestSerializer, PricingResponseSerializer,
    PortfolioAnalysisRequestSerializer, PortfolioAnalysisResponseSerializer,
    ImpliedVolatilityRequestSerializer, ImpliedVolatilityResponseSerializer,
    SensitivityAnalysisRequestSerializer, SensitivityAnalysisResponseSerializer,
    GreeksSerializer, PricingResultSerializer
)
from .services.black_scholes_service import BlackScholesService
from .services.binomial_service import BinomialService
from .services.monte_carlo_service import MonteCarloService
from .services.yahoo_finance_service import YahooFinanceService


class OptionViewSet(viewsets.ModelViewSet):
    """
    ViewSet para gestionar opciones financieras
    """
    queryset = Option.objects.all()
    
    def get_serializer_class(self):
        if self.action == 'create':
            return OptionCreateSerializer
        return OptionSerializer
    
    @action(detail=True, methods=['post'])
    def calculate_price(self, request, pk=None):
        """
        Calcula el precio de una opción usando el modelo especificado
        """
        option = self.get_object()
        serializer = PricingRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        model_name = data['model']
        calculate_greeks = data['calculate_greeks']
        
        # Crear clave de cache
        cache_key = f"pricing_{option.id}_{model_name}_{hash(str(data))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return Response(cached_result)
        
        start_time = time.time()
        
        try:
            # Seleccionar servicio de pricing
            if model_name == 'black_scholes':
                calculated_price = BlackScholesService.price(option)
                greeks_dict = BlackScholesService.greeks(option) if calculate_greeks else None
            elif model_name == 'binomial':
                n_steps = data.get('n_steps', 100)
                calculated_price = BinomialService.price(option, n_steps)
                greeks_dict = BinomialService.greeks(option, n_steps) if calculate_greeks else None
            elif model_name == 'monte_carlo':
                n_sim = data.get('n_simulations', 10000)
                n_steps = data.get('n_steps', 100)
                seed = data.get('seed')
                calculated_price = MonteCarloService.price(option, n_sim, n_steps, seed)
                greeks_dict = MonteCarloService.greeks(option, n_sim, n_steps, seed) if calculate_greeks else None
            else:
                return Response(
                    {'error': 'Modelo no soportado'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Guardar resultado en base de datos
            pricing_result = PricingResult.objects.create(
                option=option,
                model=model_name,
                calculated_price=Decimal(str(calculated_price)),
                n_steps=data.get('n_steps') if model_name != 'black_scholes' else None,
                n_simulations=data.get('n_simulations') if model_name == 'monte_carlo' else None
            )
            
            # Guardar griegas si se calcularon
            greeks_obj = None
            if greeks_dict:
                if model_name == 'black_scholes':
                    greeks_obj = BlackScholesService.save_greeks_to_db(option, greeks_dict)
                elif model_name == 'binomial':
                    greeks_obj = BinomialService.save_greeks_to_db(option, greeks_dict, data.get('n_steps', 100))
                elif model_name == 'monte_carlo':
                    greeks_obj = MonteCarloService.save_greeks_to_db(
                        option, greeks_dict, data.get('n_simulations', 10000), data.get('n_steps', 100)
                    )
            
            calculation_time = time.time() - start_time
            
            response_data = {
                'option_id': option.id,
                'model': model_name,
                'calculated_price': calculated_price,
                'implied_volatility': None,
                'greeks': GreeksSerializer(greeks_obj).data if greeks_obj else None,
                'calculation_time': calculation_time
            }
            
            # Cache result for 5 minutes
            cache.set(cache_key, response_data, 300)
            
            return Response(response_data)
            
        except Exception as e:
            return Response(
                {'error': f'Error en el cálculo: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def implied_volatility(self, request, pk=None):
        """
        Calcula la volatilidad implícita de una opción
        """
        option = self.get_object()
        serializer = ImpliedVolatilityRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        market_price = float(data['market_price'])
        max_iterations = data['max_iterations']
        tolerance = data['tolerance']
        
        try:
            start_time = time.time()
            
            # Solo Black-Scholes para volatilidad implícita por ahora
            implied_vol = BlackScholesService.implied_volatility(
                option, market_price, max_iterations, tolerance
            )
            
            # Actualizar el precio de mercado de la opción
            option.market_price = Decimal(str(market_price))
            option.volatility = Decimal(str(implied_vol))
            option.save()
            
            calculation_time = time.time() - start_time
            
            response_data = {
                'option_id': option.id,
                'market_price': market_price,
                'implied_volatility': implied_vol,
                'iterations_used': max_iterations,  # Simplificado
                'converged': True,  # Simplificado
            }
            
            return Response(response_data)
            
        except Exception as e:
            return Response(
                {'error': f'Error calculando volatilidad implícita: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def sensitivity_analysis(self, request, pk=None):
        """
        Realiza análisis de sensibilidad de una opción
        """
        option = self.get_object()
        serializer = SensitivityAnalysisRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        sensitivity_type = data['sensitivity_type']
        min_value = data['min_value']
        max_value = data['max_value']
        num_points = data['num_points']
        model_name = data['model']
        
        try:
            start_time = time.time()
            
            # Generar puntos para el análisis
            values = [min_value + i * (max_value - min_value) / (num_points - 1) 
                     for i in range(num_points)]
            
            data_points = []
            
            for value in values:
                # Crear copia temporal de la opción con el parámetro modificado
                temp_option = Option(
                    name=option.name,
                    type=option.type,
                    style=option.style,
                    spot=option.spot,
                    strike=option.strike,
                    maturity=option.maturity,
                    volatility=option.volatility,
                    rate=option.rate,
                    market_price=option.market_price
                )
                
                # Modificar el parámetro específico
                if sensitivity_type == 'spot':
                    temp_option.spot = Decimal(str(value))
                elif sensitivity_type == 'strike':
                    temp_option.strike = Decimal(str(value))
                elif sensitivity_type == 'volatility':
                    temp_option.volatility = Decimal(str(value))
                elif sensitivity_type == 'rate':
                    temp_option.rate = Decimal(str(value))
                elif sensitivity_type == 'time':
                    temp_option.maturity = Decimal(str(value))
                
                # Calcular precio
                if model_name == 'black_scholes':
                    price = BlackScholesService.price(temp_option)
                elif model_name == 'binomial':
                    price = BinomialService.price(temp_option, 100)
                elif model_name == 'monte_carlo':
                    price = MonteCarloService.price(temp_option, 10000, 100)
                
                data_points.append({
                    'parameter_value': value,
                    'option_price': price
                })
            
            calculation_time = time.time() - start_time
            
            response_data = {
                'option_id': option.id,
                'sensitivity_type': sensitivity_type,
                'model_used': model_name,
                'data_points': data_points,
                'calculation_time': calculation_time
            }
            
            return Response(response_data)
            
        except Exception as e:
            return Response(
                {'error': f'Error en análisis de sensibilidad: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PortfolioViewSet(viewsets.ModelViewSet):
    """
    ViewSet para gestionar portfolios de opciones
    """
    queryset = Portfolio.objects.all()
    
    def get_serializer_class(self):
        if self.action == 'create':
            return PortfolioCreateSerializer
        return PortfolioSerializer
    
    @action(detail=True, methods=['post'])
    def add_position(self, request, pk=None):
        """
        Añade una posición al portfolio
        """
        portfolio = self.get_object()
        serializer = PortfolioPositionSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            position = serializer.save(portfolio=portfolio)
            return Response(
                PortfolioPositionSerializer(position).data, 
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            return Response(
                {'error': f'Error añadiendo posición: {str(e)}'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['post'])
    def analyze(self, request, pk=None):
        """
        Realiza análisis completo del portfolio
        """
        portfolio = self.get_object()
        serializer = PortfolioAnalysisRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        model_name = data['model']
        calculate_var = data['calculate_var']
        confidence_level = data['confidence_level']
        horizon_days = data['horizon_days']
        n_simulations = data['n_simulations']
        n_steps = data['n_steps']
        
        try:
            start_time = time.time()
            
            # Calcular valor total del portfolio
            total_value = 0
            total_greeks = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
            
            for position in portfolio.positions.all():
                option = position.option
                quantity = float(position.quantity)
                
                # Calcular precio de la opción
                if model_name == 'black_scholes':
                    option_price = BlackScholesService.price(option)
                    greeks = BlackScholesService.greeks(option)
                elif model_name == 'binomial':
                    option_price = BinomialService.price(option, n_steps)
                    greeks = BinomialService.greeks(option, n_steps)
                elif model_name == 'monte_carlo':
                    option_price = MonteCarloService.price(option, n_simulations, n_steps)
                    greeks = MonteCarloService.greeks(option, n_simulations, n_steps)
                
                # Agregar al valor total
                total_value += option_price * quantity
                
                # Agregar griegas ponderadas
                for greek_name in total_greeks:
                    total_greeks[greek_name] += greeks[greek_name] * quantity
            
            # Calcular VaR y ES si se solicita (implementación simplificada)
            var = None
            expected_shortfall = None
            
            if calculate_var:
                # Simulación Monte Carlo para VaR del portfolio
                import numpy as np
                
                returns = []
                for _ in range(1000):  # Simulaciones para VaR
                    portfolio_value = 0
                    for position in portfolio.positions.all():
                        option = position.option
                        quantity = float(position.quantity)
                        
                        # Simular nuevo precio del subyacente
                        dt = horizon_days / 365.0
                        S = float(option.spot)
                        r = float(option.rate)
                        sigma = float(option.volatility)
                        
                        Z = np.random.normal(0, 1)
                        S_new = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
                        
                        # Crear opción temporal con nuevo precio spot
                        temp_option = Option(
                            name=option.name,
                            type=option.type,
                            style=option.style,
                            spot=Decimal(str(S_new)),
                            strike=option.strike,
                            maturity=option.maturity - Decimal(str(dt)),
                            volatility=option.volatility,
                            rate=option.rate
                        )
                        
                        # Calcular nuevo precio
                        if model_name == 'black_scholes':
                            new_price = BlackScholesService.price(temp_option)
                        elif model_name == 'binomial':
                            new_price = BinomialService.price(temp_option, n_steps)
                        elif model_name == 'monte_carlo':
                            new_price = MonteCarloService.price(temp_option, n_simulations//10, n_steps)
                        
                        portfolio_value += new_price * quantity
                    
                    returns.append(portfolio_value - total_value)
                
                # Calcular VaR y ES
                returns = np.array(returns)
                var_percentile = (1 - confidence_level) * 100
                var = -np.percentile(returns, var_percentile)
                
                # Expected Shortfall (Conditional VaR)
                es_mask = returns <= -var
                if np.any(es_mask):
                    expected_shortfall = -np.mean(returns[es_mask])
                else:
                    expected_shortfall = var
            
            calculation_time = time.time() - start_time
            
            response_data = {
                'portfolio_id': portfolio.id,
                'total_value': total_value,
                'total_greeks': total_greeks,
                'var': var,
                'expected_shortfall': expected_shortfall,
                'model_used': model_name,
                'calculation_time': calculation_time
            }
            
            return Response(response_data)
            
        except Exception as e:
            return Response(
                {'error': f'Error en análisis del portfolio: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PricingResultViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet de solo lectura para resultados de pricing
    """
    queryset = PricingResult.objects.all()
    serializer_class = PricingResultSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        option_id = self.request.query_params.get('option_id')
        model = self.request.query_params.get('model')
        
        if option_id:
            queryset = queryset.filter(option_id=option_id)
        if model:
            queryset = queryset.filter(model=model)
        
        return queryset.order_by('-calculated_at')


class HealthCheckView(APIView):
    """
    Vista simple para health check
    """
    def get(self, request):
        return Response({
            'status': 'ok',
            'service': 'option_pricing',
            'timestamp': time.time()
        })


class YahooFinanceTestView(APIView):
    """
    Vista para testear la conexión a Yahoo Finance
    """
    def get(self, request):
        """
        Ejecuta un test completo de conectividad a Yahoo Finance
        """
        try:
            test_result = YahooFinanceService.test_connection()
            
            # Determinar el status code basado en el resultado
            status_code = status.HTTP_200_OK if test_result['status'] == 'success' else status.HTTP_503_SERVICE_UNAVAILABLE
            
            return Response(test_result, status=status_code)
            
        except Exception as e:
            return Response({
                'status': 'error',
                'message': f'Error ejecutando test de conexión: {str(e)}',
                'timestamp': time.time()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class YahooFinanceTickerView(APIView):
    """
    Vista para obtener información de tickers específicos de Yahoo Finance
    """
    def get(self, request, symbol=None):
        """
        Obtiene información de un ticker específico
        """
        if not symbol:
            symbol = request.query_params.get('symbol')
        
        if not symbol:
            return Response({
                'status': 'error',
                'message': 'Debe proporcionar un símbolo de ticker'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Obtener información básica del ticker
            ticker_info = YahooFinanceService.get_ticker_info(symbol)
            
            # Si también se solicitan datos históricos
            include_history = request.query_params.get('include_history', 'false').lower() == 'true'
            if include_history and ticker_info['status'] == 'success':
                period = request.query_params.get('period', '1mo')
                historical_data = YahooFinanceService.get_historical_data(symbol, period)
                ticker_info['historical_data'] = historical_data
            
            status_code = status.HTTP_200_OK if ticker_info['status'] == 'success' else status.HTTP_404_NOT_FOUND
            
            return Response(ticker_info, status=status_code)
            
        except Exception as e:
            return Response({
                'status': 'error',
                'message': f'Error obteniendo información del ticker {symbol}: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class YahooFinanceOptionsView(APIView):
    """
    Vista para obtener información de opciones de Yahoo Finance
    """
    
    def get(self, request):
        """
        Obtiene las fechas de expiración disponibles para un símbolo
        """
        symbol = request.GET.get('symbol', '').upper()
        
        if not symbol:
            return Response(
                {'error': 'Símbolo requerido'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            result = YahooFinanceService.get_options_expirations(symbol)
            return Response(result)
        except Exception as e:
            return Response(
                {'error': f'Error obteniendo expiraciones: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def post(self, request):
        """
        Obtiene la cadena de opciones para una fecha específica
        """
        symbol = request.data.get('symbol', '').upper()
        expiration_date = request.data.get('expiration_date', '')
        option_type = request.data.get('option_type', 'call')
        
        if not symbol or not expiration_date:
            return Response(
                {'error': 'Símbolo y fecha de expiración requeridos'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            if option_type == 'atm':
                # Obtener opciones ATM
                result = YahooFinanceService.get_atm_options(symbol, expiration_date, request.data.get('type', 'call'))
            else:
                # Obtener cadena completa
                result = YahooFinanceService.get_options_chain(symbol, expiration_date)
            
            return Response(result)
        except Exception as e:
            return Response(
                {'error': f'Error obteniendo opciones: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
