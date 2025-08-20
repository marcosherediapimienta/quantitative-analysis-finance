# Configuración del proyecto Quantitative Finance
# Este archivo contiene configuraciones específicas del proyecto

# Configuración de la API
API_VERSION = 'v1'
API_TITLE = 'Quantitative Finance API'
API_DESCRIPTION = 'API para análisis cuantitativo de finanzas'

# Configuración de aplicaciones
INSTALLED_APPS = [
    'apps.option_pricing',
    'apps.portfolio_management',
]

# Configuración de URLs de la API
API_URLS = {
    'option_pricing': 'api/option-pricing/',
    'portfolio_management': 'api/portfolio-management/',
}

# Configuración de servicios externos
YAHOO_FINANCE_BASE_URL = 'https://query1.finance.yahoo.com'
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co'

# Configuración de límites
MAX_PORTFOLIO_SIZE = 100
MAX_HISTORICAL_DAYS = 2520  # 10 años de datos históricos

# Configuración de cache
CACHE_TIMEOUT = 3600  # 1 hora en segundos
REDIS_CACHE_KEY_PREFIX = 'quant_finance'

# Configuración de logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
