import yfinance as yf
import requests
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """
    Servicio para interactuar con Yahoo Finance API usando yfinance
    """
    
    @staticmethod
    def test_connection() -> Dict[str, Any]:
        """
        Prueba la conexión con Yahoo Finance
        """
        start_time = time.time()
        result = {
            'status': 'success',
            'message': 'Conexión exitosa con Yahoo Finance',
            'response_time': 0,
            'test_details': []
        }
        
        try:
            # Test 1: Crear ticker
            result['test_details'].append('✓ Creando ticker de prueba...')
            ticker = yf.Ticker('AAPL')
            
            # Test 2: Obtener información básica
            result['test_details'].append('✓ Obteniendo información básica...')
            info = ticker.info
            if not info:
                raise Exception('No se pudo obtener información del ticker')
            
            # Test 3: Obtener precio actual
            result['test_details'].append('✓ Obteniendo precio actual...')
            current_price = info.get('currentPrice')
            if current_price is None:
                raise Exception('No se pudo obtener precio actual')
            
            # Test 4: Obtener expiraciones de opciones
            result['test_details'].append('✓ Obteniendo expiraciones de opciones...')
            expirations = ticker.options
            if not expirations:
                result['test_details'].append('⚠ No se encontraron opciones disponibles para AAPL')
            else:
                result['test_details'].append(f'✓ Encontradas {len(expirations)} expiraciones')
            
            # Test 5: Obtener cadena de opciones (si hay expiraciones)
            if expirations:
                try:
                    result['test_details'].append('✓ Probando obtención de cadena de opciones...')
                    options = ticker.option_chain(expirations[0])
                    if hasattr(options, 'calls') or hasattr(options, 'puts'):
                        result['test_details'].append('✓ Cadena de opciones obtenida correctamente')
                    else:
                        result['test_details'].append('⚠ Estructura de opciones inesperada')
                except Exception as e:
                    result['test_details'].append(f'⚠ Error obteniendo cadena de opciones: {str(e)}')
            
            response_time = time.time() - start_time
            result['response_time'] = round(response_time, 3)
            result['test_details'].append(f'✓ Test completado en {result["response_time"]}s')
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f'Error en test de conexión: {str(e)}'
            
            result.update({
                'status': 'error',
                'message': error_msg,
                'response_time': response_time
            })
            result['test_details'].append(f'✗ {error_msg}')
            
            logger.error(f"Error en test de Yahoo Finance: {str(e)}")
        
        return result
    
    @staticmethod
    def get_ticker_info(symbol: str) -> Dict[str, Any]:
        """
        Obtiene información básica de un ticker específico
        
        Args:
            symbol: Símbolo del ticker (ej: 'AAPL', 'MSFT')
            
        Returns:
            Dict con información del ticker o error
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            if not info:
                return {
                    'status': 'error',
                    'message': f'No se encontró información para el símbolo {symbol}'
                }
            
            return {
                'status': 'success',
                'symbol': symbol.upper(),
                'data': {
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'current_price': info.get('currentPrice', 'N/A'),
                    'previous_close': info.get('previousClose', 'N/A'),
                    'market_cap': info.get('marketCap', 'N/A'),
                    'currency': info.get('currency', 'USD')
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error obteniendo información de {symbol}: {str(e)}'
            }
    
    @staticmethod
    def get_historical_data(symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """
        Obtiene datos históricos de un ticker
        
        Args:
            symbol: Símbolo del ticker
            period: Período de datos ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            
        Returns:
            Dict con datos históricos o error
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                return {
                    'status': 'error',
                    'message': f'No se encontraron datos históricos para {symbol}'
                }
            
            # Convertir datos a formato serializable
            data_points = []
            for date, row in hist_data.iterrows():
                data_points.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']) if 'Volume' in row else 0
                })
            
            return {
                'status': 'success',
                'symbol': symbol.upper(),
                'period': period,
                'data_points': len(data_points),
                'data': data_points
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error obteniendo datos históricos de {symbol}: {str(e)}'
            }

    @staticmethod
    def get_options_expirations(symbol: str) -> Dict[str, Any]:
        """
        Obtiene las fechas de expiración disponibles para opciones de un ticker
        
        Args:
            symbol: Símbolo del ticker (ej: 'AAPL', 'MSFT')
            
        Returns:
            Dict con fechas de expiración disponibles
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            expirations = ticker.options
            
            if not expirations:
                return {
                    'status': 'error',
                    'message': f'No se encontraron opciones disponibles para {symbol}'
                }
            
            # Formatear fechas
            formatted_expirations = []
            for exp in expirations:
                try:
                    date_obj = datetime.strptime(exp, '%Y-%m-%d')
                    formatted_expirations.append({
                        'date': exp,
                        'formatted': date_obj.strftime('%d/%m/%Y'),
                        'days_to_expiry': (date_obj - datetime.now()).days
                    })
                except:
                    formatted_expirations.append({
                        'date': exp,
                        'formatted': exp,
                        'days_to_expiry': 0
                    })
            
            return {
                'status': 'success',
                'symbol': symbol.upper(),
                'expirations': formatted_expirations,
                'count': len(formatted_expirations)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error obteniendo expiraciones de opciones para {symbol}: {str(e)}'
            }
    
    @staticmethod
    def get_options_chain(symbol: str, expiration_date: str) -> Dict[str, Any]:
        """
        Obtiene la cadena de opciones para una fecha de expiración específica
        
        Args:
            symbol: Símbolo del ticker
            expiration_date: Fecha de expiración en formato 'YYYY-MM-DD'
            
        Returns:
            Dict con calls y puts disponibles
        """
        try:
            logger.info(f"Obteniendo cadena de opciones para {symbol} en {expiration_date}")
            ticker = yf.Ticker(symbol.upper())
            
            # Obtener opciones para la fecha específica
            logger.info(f"Llamando a ticker.option_chain({expiration_date})")
            options = ticker.option_chain(expiration_date)
            
            if not options:
                logger.warning(f"No se encontraron opciones para {symbol} en {expiration_date}")
                return {
                    'status': 'error',
                    'message': f'No se encontraron opciones para {symbol} en {expiration_date}'
                }
            
            logger.info(f"Opciones obtenidas para {symbol} en {expiration_date}")
            
            # Función auxiliar para convertir valores de manera segura
            def safe_float(value, default=0.0):
                """Convierte un valor a float de manera segura, manejando NaN"""
                if value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=0):
                """Convierte un valor a int de manera segura, manejando NaN"""
                if value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                    return default
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return default
            
            # Procesar calls
            calls = []
            if hasattr(options, 'calls') and options.calls is not None:
                logger.info(f"Procesando {len(options.calls)} calls para {symbol}")
                for i, call in enumerate(options.calls.itertuples()):
                    try:
                        call_data = {
                            'strike': safe_float(call.strike),
                            'last_price': safe_float(call.lastPrice),
                            'bid': safe_float(call.bid),
                            'ask': safe_float(call.ask),
                            'volume': safe_int(call.volume),
                            'open_interest': safe_int(call.openInterest),
                            'implied_volatility': safe_float(call.impliedVolatility)
                        }
                        calls.append(call_data)
                    except Exception as e:
                        logger.warning(f"Error procesando call {i}: {str(e)}")
                        continue
            else:
                logger.info(f"No se encontraron calls para {symbol}")
            
            # Procesar puts
            puts = []
            if hasattr(options, 'puts') and options.puts is not None:
                logger.info(f"Procesando {len(options.puts)} puts para {symbol}")
                for i, put in enumerate(options.puts.itertuples()):
                    try:
                        put_data = {
                            'strike': safe_float(put.strike),
                            'last_price': safe_float(put.lastPrice),
                            'bid': safe_float(put.bid),
                            'ask': safe_float(put.ask),
                            'volume': safe_int(put.volume),
                            'open_interest': safe_int(put.openInterest),
                            'implied_volatility': safe_float(put.impliedVolatility)
                        }
                        puts.append(put_data)
                    except Exception as e:
                        logger.warning(f"Error procesando put {i}: {str(e)}")
                        continue
            else:
                logger.info(f"No se encontraron puts para {symbol}")
            
            # Obtener precio actual del subyacente
            current_price = safe_float(ticker.info.get('currentPrice', 0))
            logger.info(f"Precio actual de {symbol}: ${current_price}")
            
            result = {
                'status': 'success',
                'symbol': symbol.upper(),
                'expiration_date': expiration_date,
                'current_price': current_price,
                'calls': calls,
                'puts': puts,
                'total_calls': len(calls),
                'total_puts': len(puts)
            }
            
            logger.info(f"Cadena de opciones procesada exitosamente: {len(calls)} calls, {len(puts)} puts")
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo cadena de opciones para {symbol} en {expiration_date}: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error obteniendo cadena de opciones para {symbol} en {expiration_date}: {str(e)}'
            }
    
    @staticmethod
    def get_atm_options(symbol: str, expiration_date: str, option_type: str = 'call') -> Dict[str, Any]:
        """
        Obtiene opciones at-the-money para análisis
        
        Args:
            symbol: Símbolo del ticker
            expiration_date: Fecha de expiración
            option_type: Tipo de opción ('call' o 'put')
            
        Returns:
            Dict con opciones ATM más relevantes
        """
        try:
            chain_data = YahooFinanceService.get_options_chain(symbol, expiration_date)
            
            if chain_data['status'] != 'success':
                return chain_data
            
            current_price = chain_data['current_price']
            options = chain_data['calls'] if option_type == 'call' else chain_data['puts']
            
            if not options:
                return {
                    'status': 'error',
                    'message': f'No se encontraron opciones {option_type} para {symbol}'
                }
            
            # Encontrar opciones ATM (strike más cercano al precio actual)
            atm_options = []
            for option in options:
                strike_diff = abs(option['strike'] - current_price)
                atm_options.append({
                    **option,
                    'strike_diff': strike_diff
                })
            
            # Ordenar por proximidad al precio actual
            atm_options.sort(key=lambda x: x['strike_diff'])
            
            # Tomar las 5 opciones más ATM
            best_atm = atm_options[:5]
            
            return {
                'status': 'success',
                'symbol': symbol.upper(),
                'expiration_date': expiration_date,
                'option_type': option_type,
                'current_price': current_price,
                'atm_options': best_atm
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error obteniendo opciones ATM para {symbol}: {str(e)}'
            }
