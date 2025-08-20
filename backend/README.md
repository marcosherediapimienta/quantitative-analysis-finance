# Backend Django - Quantitative Finance

## 🏗️ Estructura del Proyecto

```
backend/
│── apps/                          # Aplicaciones Django
│   ├── option_pricing/           # Precios de opciones
│   ├── portfolio_management/     # Gestión de portfolios
│   └── quantitative_finance/     # Configuración principal
│── core/                         # Configuración central
│   ├── settings.py              # Configuraciones de Django
│   ├── urls.py                  # URLs principales
│   ├── wsgi.py                  # Configuración WSGI
│   └── asgi.py                  # Configuración ASGI
│── scripts/                      # Scripts de utilidad
│   ├── install.sh               # Instalación del proyecto
│   ├── setup_django.sh          # Configuración de Django
│   └── start.sh                 # Inicio del servidor
│── docker/                       # Configuración Docker
│   ├── Dockerfile               # Imagen Docker
│   └── docker-compose.yml       # Orquestación de servicios
│── config/                       # Configuraciones
│   ├── env_example.txt          # Variables de entorno de ejemplo
│   └── project_config.py        # Configuración del proyecto
│── requirements.txt              # Dependencias de Python
└── manage.py                     # Utilidad de gestión de Django
```

## 🚀 Inicio Rápido

### 1. Instalación
```bash
# Clonar el repositorio
git clone <repository-url>
cd backend

# Ejecutar script de instalación
chmod +x scripts/install.sh
./scripts/install.sh
```

### 2. Configuración
```bash
# Copiar archivo de variables de entorno
cp config/env_example.txt .env

# Editar variables de entorno según tu configuración
nano .env
```

### 3. Inicio del Servidor
```bash
# Ejecutar script de inicio
chmod +x scripts/start.sh
./scripts/start.sh
```

## 🐳 Docker

### Desarrollo
```bash
cd docker
docker-compose up --build
```

### Producción
```bash
cd docker
docker-compose -f docker-compose.prod.yml up --build
```

## 🔧 Desarrollo

### Estructura de Aplicaciones
- **option_pricing**: Modelos y servicios para precios de opciones
  - Black-Scholes, Binomial, Monte Carlo
  - Cálculo de griegas (Delta, Gamma, Vega, Theta, Rho)
  - Volatilidad implícita y análisis de sensibilidad
- **portfolio_management**: Gestión y optimización de portfolios
  - Análisis técnico y fundamental
  - Métricas de riesgo y optimización

### Comandos Útiles
```bash
# Crear migraciones
python manage.py makemigrations

# Aplicar migraciones
python manage.py migrate

# Crear superusuario
python manage.py createsuperuser

# Recopilar archivos estáticos
python manage.py collectstatic

# Ejecutar tests
python manage.py test

# Verificar configuración
python manage.py check
```

## 🔗 URLs de la API

- **Admin**: http://127.0.0.1:8000/admin/
- **API Option Pricing**: http://127.0.0.1:8000/api/option-pricing/
- **API Portfolio Management**: http://127.0.0.1:8000/api/portfolio-management/
- **Health Check**: http://127.0.0.1:8000/api/option-pricing/health/

## 📦 Dependencias Principales

- **Django 4.2+** - Framework web
- **Django REST Framework** - API REST
- **Librerías científicas** - numpy, pandas, scipy, matplotlib
- **Librerías financieras** - yfinance, finta, QuantLib
- **Análisis técnico** - ta-lib, scikit-learn, cvxpy
- **Herramientas de desarrollo** - pytest, black, flake8

## ⚙️ Configuración

### Variables de Entorno (.env)
```bash
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
REDIS_URL=redis://localhost:6379
```

### Base de Datos
- **Desarrollo**: SQLite (por defecto)
- **Producción**: PostgreSQL (configurable)

## 🧪 Testing

```bash
# Ejecutar todos los tests
python manage.py test

# Tests específicos
python manage.py test apps.option_pricing
python manage.py test apps.portfolio_management

# Con coverage
coverage run --source='.' manage.py test
coverage report
```

## 📊 Funcionalidades Implementadas

### Option Pricing
- ✅ Pricing con Black-Scholes, Binomial y Monte Carlo
- ✅ Cálculo de griegas completas
- ✅ Volatilidad implícita
- ✅ Análisis de sensibilidad paramétrico
- ✅ Gestión de portfolios de opciones
- ✅ Análisis de riesgo (VaR, Expected Shortfall)

### Portfolio Management
- ✅ Modelos para análisis técnico
- ✅ Modelos para análisis fundamental
- ✅ Gestión de datos de mercado
- ✅ Métricas de riesgo
- ✅ Optimización de portfolios

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
