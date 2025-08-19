import yfinance as yf
import requests
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """
    Servicio para testear y obtener datos de Yahoo Finance
    """
    
    @staticmethod
    def test_connection() -> Dict[str, Any]:
        """
        Testea la conexión a Yahoo Finance
        
        Returns:
            Dict con el resultado del test incluyendo:
            - status: 'success' o 'error'
            - message: descripción del resultado
            - response_time: tiempo de respuesta en segundos
            - data_sample: muestra de datos obtenidos si es exitoso
        """
        result = {
            'status': 'error',
            'message': '',
            'response_time': None,
            'data_sample': None,
            'test_timestamp': datetime.now().isoformat(),
            'test_details': []
        }
        
        start_time = datetime.now()
        
        try:
            # Test 1: Verificar conectividad básica a Yahoo Finance
            logger.info("Iniciando test de conexión a Yahoo Finance...")
            result['test_details'].append("Verificando conectividad básica...")
            
            # Intentar conexión directa al dominio con reintentos
            max_retries = 3
            retry_delay = 2  # segundos
            
            for attempt in range(max_retries):
                try:
                    response = requests.get('https://finance.yahoo.com', timeout=10)
                    if response.status_code == 200:
                        result['test_details'].append("✓ Conectividad al dominio: OK")
                        break
                    elif response.status_code == 429:
                        if attempt < max_retries - 1:
                            result['test_details'].append(f"⚠ Intento {attempt + 1}: Rate limit (429), reintentando en {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Backoff exponencial
                            continue
                        else:
                            result['test_details'].append("✗ Conectividad al dominio: Rate limit persistente (429)")
                            result['test_details'].append("⚠ Continuando test usando solo yfinance...")
                            break
                    else:
                        result['test_details'].append(f"✗ Conectividad al dominio: Error {response.status_code}")
                        result['test_details'].append("⚠ Continuando test usando solo yfinance...")
                        break
                except requests.RequestException as e:
                    if attempt < max_retries - 1:
                        result['test_details'].append(f"⚠ Intento {attempt + 1}: Error de red, reintentando en {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        result['test_details'].append(f"⚠ Error de red: {str(e)}")
                        result['test_details'].append("⚠ Continuando test usando solo yfinance...")
                        break
            
            # Test 2: Probar descarga de datos usando yfinance
            result['test_details'].append("Probando descarga de datos con yfinance...")
            
            # Usar un símbolo conocido y estable como AAPL
            # Si hay rate limit, intentar con un ticker diferente
            ticker_symbols = ["AAPL", "MSFT", "GOOGL"]
            ticker = None
            ticker_symbol = None
            
            for symbol in ticker_symbols:
                try:
                    temp_ticker = yf.Ticker(symbol)
                    # Hacer una petición simple para verificar
                    info = temp_ticker.info
                    if info and len(info) > 0:
                        ticker = temp_ticker
                        ticker_symbol = symbol
                        result['test_details'].append(f"✓ Usando ticker alternativo: {symbol}")
                        break
                except Exception as e:
                    result['test_details'].append(f"⚠ Error con ticker {symbol}: {str(e)}")
                    continue
            
            if not ticker:
                result['test_details'].append("✗ No se pudo obtener ningún ticker válido")
                result['message'] = "Error: No se pudo obtener datos de ningún ticker"
                return result
            
            # Obtener información básica
            info = ticker.info
            if not info or len(info) == 0:
                result['test_details'].append("✗ No se pudo obtener información básica del ticker")
                result['message'] = "Error: No se pudo obtener información del ticker"
                return result
            
            result['test_details'].append(f"✓ Información básica obtenida para {ticker_symbol}")
            
            # Test 3: Obtener datos históricos recientes
            result['test_details'].append("Obteniendo datos históricos...")
            
            # Obtener datos de los últimos 5 días
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # 7 días para asegurar que tenemos datos
            
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                result['test_details'].append("✗ No se pudieron obtener datos históricos")
                result['message'] = "Error: No se pudieron obtener datos históricos"
                return result
            
            result['test_details'].append(f"✓ Datos históricos obtenidos: {len(hist_data)} registros")
            
            # Test 4: Verificar que los datos son válidos
            result['test_details'].append("Validando datos obtenidos...")
            
            if 'Close' not in hist_data.columns:
                result['test_details'].append("✗ Datos históricos no contienen precio de cierre")
                result['message'] = "Error: Estructura de datos inválida"
                return result
            
            latest_price = hist_data['Close'].iloc[-1]
            if not latest_price or latest_price <= 0:
                result['test_details'].append("✗ Precio de cierre inválido")
                result['message'] = "Error: Precio de cierre inválido"
                return result
            
            result['test_details'].append(f"✓ Precio de cierre válido: ${latest_price:.2f}")
            
            # Calcular tiempo de respuesta
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Preparar muestra de datos para la respuesta
            data_sample = {
                'ticker': ticker_symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'latest_price': float(latest_price),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', 'N/A'),
                'data_points_retrieved': len(hist_data),
                'date_range': {
                    'start': hist_data.index[0].strftime('%Y-%m-%d') if len(hist_data) > 0 else None,
                    'end': hist_data.index[-1].strftime('%Y-%m-%d') if len(hist_data) > 0 else None
                }
            }
            
            # Test exitoso
            result.update({
                'status': 'success',
                'message': 'Conexión a Yahoo Finance exitosa',
                'response_time': response_time,
                'data_sample': data_sample
            })
            
            result['test_details'].append(f"✓ Test completado exitosamente en {response_time:.2f} segundos")
            
            logger.info(f"Test de Yahoo Finance completado exitosamente en {response_time:.2f}s")
            
        except Exception as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            error_msg = f"Error inesperado durante el test: {str(e)}"
            result.update({
                'status': 'error',
                'message': error_msg,
                'response_time': response_time
            })
            result['test_details'].append(f"✗ {error_msg}")
            
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
