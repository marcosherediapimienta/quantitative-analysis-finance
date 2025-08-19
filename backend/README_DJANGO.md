# Backend Django para Quantitative Finance

Este proyecto ha sido convertido de FastAPI a Django REST Framework para proporcionar una API robusta y escalable para análisis cuantitativo financiero.

## 🚀 Configuración Inicial

### Prerrequisitos
- Python 3.9+
- pip
- virtualenv (recomendado)

### Instalación

1. **Configurar el proyecto automáticamente:**
   ```bash
   chmod +x setup_django.sh
   ./setup_django.sh
   ```

2. **O configurar manualmente:**
   ```bash
   # Crear entorno virtual
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   
   # Instalar dependencias
   pip install -r requirements-django.txt
   
   # Configurar base de datos
   python manage.py makemigrations
   python manage.py migrate
   
   # Crear superusuario
   python manage.py createsuperuser
   
   # Ejecutar servidor
   python manage.py runserver
   ```

## 📁 Estructura del Proyecto

```
quantitative-analysis-finance/
├── manage.py                          # Comando de gestión de Django
├── quantitative_finance/              # Configuración principal del proyecto
│   ├── __init__.py
│   ├── settings.py                    # Configuración de Django
│   ├── urls.py                        # URLs principales
│   ├── wsgi.py                        # Configuración WSGI
│   └── asgi.py                        # Configuración ASGI
├── option_pricing/                    # Aplicación de pricing de opciones
│   ├── models.py                      # Modelos de datos
│   ├── serializers.py                # Serializers para API
│   ├── views.py                       # Vistas de API
│   ├── urls.py                        # URLs de la aplicación
│   ├── admin.py                       # Configuración del admin
│   └── services/                      # Servicios de cálculo
│       ├── black_scholes_service.py
│       ├── binomial_service.py
│       └── monte_carlo_service.py
├── portfolio_management/              # Aplicación de gestión de portfolios
│   ├── models.py                      # Modelos para análisis de portfolio
│   ├── urls.py                        # URLs de la aplicación
│   └── admin.py                       # Configuración del admin
└── requirements-django.txt            # Dependencias de Django
```

## 🔗 Endpoints de API

### Option Pricing API (`/api/option-pricing/`)

#### Opciones
- `GET /api/option-pricing/options/` - Listar todas las opciones
- `POST /api/option-pricing/options/` - Crear nueva opción
- `GET /api/option-pricing/options/{id}/` - Obtener opción específica
- `PUT /api/option-pricing/options/{id}/` - Actualizar opción
- `DELETE /api/option-pricing/options/{id}/` - Eliminar opción

#### Pricing
- `POST /api/option-pricing/options/{id}/calculate_price/` - Calcular precio de opción
- `POST /api/option-pricing/options/{id}/implied_volatility/` - Calcular volatilidad implícita
- `POST /api/option-pricing/options/{id}/sensitivity_analysis/` - Análisis de sensibilidad

#### Portfolios
- `GET /api/option-pricing/portfolios/` - Listar portfolios
- `POST /api/option-pricing/portfolios/` - Crear portfolio
- `POST /api/option-pricing/portfolios/{id}/add_position/` - Añadir posición
- `POST /api/option-pricing/portfolios/{id}/analyze/` - Analizar portfolio

### Portfolio Management API (`/api/portfolio-management/`)
- En desarrollo - estructura preparada para análisis técnico y fundamental

## 📊 Ejemplos de Uso

### 1. Crear una Opción
```python
import requests

# Crear una opción call europea
option_data = {
    "name": "AAPL Call Option",
    "type": "call",
    "style": "european",
    "spot": "150.00",
    "strike": "155.00",
    "maturity": "0.25",  # 3 meses
    "volatility": "0.20",
    "rate": "0.05",
    "market_price": "5.50"
}

response = requests.post(
    "http://127.0.0.1:8000/api/option-pricing/options/",
    json=option_data
)
option = response.json()
print(f"Opción creada con ID: {option['id']}")
```

### 2. Calcular Precio usando Black-Scholes
```python
pricing_request = {
    "model": "black_scholes",
    "calculate_greeks": True
}

response = requests.post(
    f"http://127.0.0.1:8000/api/option-pricing/options/{option['id']}/calculate_price/",
    json=pricing_request
)
result = response.json()
print(f"Precio calculado: {result['calculated_price']}")
print(f"Delta: {result['greeks']['delta']}")
```

### 3. Análisis de Sensibilidad
```python
sensitivity_request = {
    "sensitivity_type": "spot",
    "min_value": 140,
    "max_value": 160,
    "num_points": 21,
    "model": "black_scholes"
}

response = requests.post(
    f"http://127.0.0.1:8000/api/option-pricing/options/{option['id']}/sensitivity_analysis/",
    json=sensitivity_request
)
sensitivity_data = response.json()
```

### 4. Crear y Analizar Portfolio
```python
# Crear portfolio
portfolio_data = {
    "name": "Portfolio de Estrategia Long Straddle",
    "description": "Estrategia de volatilidad con calls y puts"
}

portfolio_response = requests.post(
    "http://127.0.0.1:8000/api/option-pricing/portfolios/",
    json=portfolio_data
)
portfolio = portfolio_response.json()

# Añadir posiciones
position_data = {
    "option_id": option['id'],
    "quantity": "10",
    "entry_price": "5.50"
}

requests.post(
    f"http://127.0.0.1:8000/api/option-pricing/portfolios/{portfolio['id']}/add_position/",
    json=position_data
)

# Analizar portfolio
analysis_request = {
    "model": "black_scholes",
    "calculate_var": True,
    "confidence_level": 0.95,
    "horizon_days": 1
}

analysis_response = requests.post(
    f"http://127.0.0.1:8000/api/option-pricing/portfolios/{portfolio['id']}/analyze/",
    json=analysis_request
)
analysis = analysis_response.json()
print(f"Valor total del portfolio: {analysis['total_value']}")
print(f"VaR al 95%: {analysis['var']}")
```

## 🔧 Configuración

### Variables de Entorno
Crea un archivo `.env` en el directorio raíz:

```bash
# Django
SECRET_KEY=tu-clave-secreta-muy-segura
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# CORS
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Base de datos (opcional - por defecto usa SQLite)
DATABASE_URL=postgresql://user:password@localhost/dbname

# Cache Redis (opcional)
REDIS_URL=redis://localhost:6379/0
```

### Configuración de Producción
Para producción, asegúrate de:
1. Configurar `DEBUG=False`
2. Usar una base de datos robusta (PostgreSQL)
3. Configurar un servidor web (nginx + gunicorn)
4. Configurar variables de entorno de seguridad

## 📈 Modelos Soportados

### Pricing de Opciones
- **Black-Scholes**: Para opciones europeas
- **Binomial**: Para opciones europeas y americanas
- **Monte Carlo**: Para opciones complejas con método Longstaff-Schwartz

### Griegas Calculadas
- Delta: Sensibilidad al precio del subyacente
- Gamma: Sensibilidad del delta
- Vega: Sensibilidad a la volatilidad
- Theta: Decay temporal
- Rho: Sensibilidad a la tasa de interés

### Análisis de Riesgo
- Value at Risk (VaR)
- Expected Shortfall (ES)
- Análisis de sensibilidad paramétrico

## 🎛️ Panel de Administración

Accede al panel de administración en `http://127.0.0.1:8000/admin/` con las credenciales de superusuario para:
- Gestionar opciones y portfolios
- Ver resultados de pricing
- Administrar datos de mercado
- Monitorear métricas de riesgo

## 🧪 Testing

```bash
# Ejecutar tests
python manage.py test

# Con coverage
pip install coverage
coverage run --source='.' manage.py test
coverage report
```

## 📝 Notas de Migración

### Diferencias con FastAPI
1. **Estructura**: Organización en aplicaciones Django
2. **ORM**: Uso de Django ORM en lugar de modelos Pydantic
3. **Admin**: Panel de administración incluido
4. **Serialización**: Django REST Framework serializers
5. **URLs**: Sistema de rutas de Django
6. **Autenticación**: Sistema de autenticación Django (extensible)

### Ventajas de Django
- ORM robusto y maduro
- Panel de administración automático
- Sistema de migraciones
- Escalabilidad y estabilidad probadas
- Extenso ecosistema de packages
- Mejor para aplicaciones empresariales

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.
