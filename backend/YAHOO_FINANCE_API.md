# API de Conexión a Yahoo Finance

## Descripción

Esta API proporciona endpoints para testear la conexión a Yahoo Finance y obtener datos de mercado en tiempo real.

## Endpoints Disponibles

### 1. Test de Conexión a Yahoo Finance

**Endpoint:** `GET /api/option-pricing/yahoo-finance/test/`

**Descripción:** Ejecuta un test completo de conectividad a Yahoo Finance para verificar que el servicio está funcionando correctamente.

**Respuesta exitosa (200):**
```json
{
    "status": "success",
    "message": "Conexión a Yahoo Finance exitosa",
    "response_time": 2.45,
    "data_sample": {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "latest_price": 175.43,
        "currency": "USD",
        "market_cap": 2750000000000,
        "data_points_retrieved": 5,
        "date_range": {
            "start": "2024-01-10",
            "end": "2024-01-15"
        }
    },
    "test_timestamp": "2024-01-15T10:30:45.123456",
    "test_details": [
        "Verificando conectividad básica...",
        "✓ Conectividad al dominio: OK",
        "Probando descarga de datos con yfinance...",
        "✓ Información básica obtenida para AAPL",
        "Obteniendo datos históricos...",
        "✓ Datos históricos obtenidos: 5 registros",
        "Validando datos obtenidos...",
        "✓ Precio de cierre válido: $175.43",
        "✓ Test completado exitosamente en 2.45 segundos"
    ]
}
```

**Respuesta de error (503):**
```json
{
    "status": "error",
    "message": "Error de conectividad: 404",
    "response_time": 1.23,
    "test_timestamp": "2024-01-15T10:30:45.123456",
    "test_details": [
        "Verificando conectividad básica...",
        "✗ Conectividad al dominio: Error 404"
    ]
}
```

### 2. Información de Ticker Específico

**Endpoint:** `GET /api/option-pricing/yahoo-finance/ticker/<symbol>/`

**Parámetros de URL:**
- `symbol`: Símbolo del ticker (ej: AAPL, MSFT, GOOGL)

**Parámetros de consulta opcionales:**
- `include_history`: `true` para incluir datos históricos (default: `false`)
- `period`: Período de datos históricos (`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`) (default: `1mo`)

**Ejemplos de uso:**

1. **Información básica:**
   ```
   GET /api/option-pricing/yahoo-finance/ticker/AAPL/
   ```

2. **Con datos históricos:**
   ```
   GET /api/option-pricing/yahoo-finance/ticker/AAPL/?include_history=true&period=1mo
   ```

3. **Usando query parameters:**
   ```
   GET /api/option-pricing/yahoo-finance/ticker/?symbol=AAPL&include_history=true
   ```

**Respuesta exitosa (200):**
```json
{
    "status": "success",
    "symbol": "AAPL",
    "data": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "current_price": 175.43,
        "previous_close": 174.50,
        "market_cap": 2750000000000,
        "currency": "USD"
    },
    "historical_data": {
        "status": "success",
        "symbol": "AAPL",
        "period": "1mo",
        "data_points": 22,
        "data": [
            {
                "date": "2024-01-01",
                "open": 174.25,
                "high": 176.80,
                "low": 173.90,
                "close": 175.43,
                "volume": 45234567
            }
        ]
    }
}
```

**Respuesta de error (404):**
```json
{
    "status": "error",
    "message": "No se encontró información para el símbolo INVALID"
}
```

## Códigos de Estado HTTP

- **200 OK**: Operación exitosa
- **400 Bad Request**: Parámetros inválidos
- **404 Not Found**: Ticker no encontrado
- **500 Internal Server Error**: Error interno del servidor
- **503 Service Unavailable**: Yahoo Finance no está disponible

## Ejemplos de Uso con curl

### Test de conexión:
```bash
curl -X GET http://localhost:8000/api/option-pricing/yahoo-finance/test/ \
     -H "Content-Type: application/json"
```

### Información de ticker:
```bash
curl -X GET http://localhost:8000/api/option-pricing/yahoo-finance/ticker/AAPL/ \
     -H "Content-Type: application/json"
```

### Ticker con datos históricos:
```bash
curl -X GET "http://localhost:8000/api/option-pricing/yahoo-finance/ticker/AAPL/?include_history=true&period=1mo" \
     -H "Content-Type: application/json"
```

## Casos de Uso

### 1. Verificación de Conectividad
Antes de realizar operaciones que dependan de Yahoo Finance, puedes usar el endpoint de test para verificar que el servicio está disponible:

```javascript
// JavaScript example
const response = await fetch('/api/option-pricing/yahoo-finance/test/');
const result = await response.json();

if (result.status === 'success') {
    console.log('Yahoo Finance está disponible');
    console.log(`Tiempo de respuesta: ${result.response_time}s`);
} else {
    console.error('Yahoo Finance no está disponible:', result.message);
}
```

### 2. Obtención de Datos de Mercado
Para obtener datos actuales de un ticker específico:

```python
# Python example
import requests

response = requests.get('http://localhost:8000/api/option-pricing/yahoo-finance/ticker/AAPL/')
data = response.json()

if data['status'] == 'success':
    price = data['data']['current_price']
    print(f"Precio actual de AAPL: ${price}")
```

### 3. Análisis con Datos Históricos
Para realizar análisis técnico con datos históricos:

```python
# Python example
import requests

response = requests.get(
    'http://localhost:8000/api/option-pricing/yahoo-finance/ticker/AAPL/',
    params={'include_history': 'true', 'period': '3mo'}
)
data = response.json()

if data['status'] == 'success':
    historical = data['historical_data']['data']
    prices = [point['close'] for point in historical]
    print(f"Obtenidos {len(prices)} puntos de precio para análisis")
```

## Registro de Logs

El servicio registra automáticamente:
- Intentos de conexión exitosos y fallidos
- Tiempos de respuesta
- Errores específicos de conectividad
- Información de debugging

Los logs se pueden encontrar en `/logs/django.log` o en la consola durante el desarrollo.

## Consideraciones de Rendimiento

- Los resultados pueden ser cacheados por Yahoo Finance
- El test de conexión utiliza AAPL como ticker de prueba por su alta liquidez
- Los datos históricos pueden tardar más en cargarse para períodos largos
- Se recomienda implementar caching a nivel de aplicación para uso intensivo

## Troubleshooting

### Error: "No se pudo obtener información del ticker"
- Verificar que el símbolo del ticker es válido
- Comprobar conectividad a internet
- Ejecutar el test de conexión para diagnosticar

### Error: "Error de red"
- Verificar conectividad a internet
- Comprobar firewall y proxy settings
- Yahoo Finance puede estar temporalmente inaccesible

### Error: "Tiempo de respuesta lento"
- Yahoo Finance puede estar experimentando alta carga
- Considerar implementar timeouts más largos
- Verificar la latencia de red
