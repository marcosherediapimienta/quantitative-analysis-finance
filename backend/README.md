# 🐍 Backend Django - Quantitative Finance

Este directorio contiene todo el backend Django para el sistema de análisis cuantitativo financiero.

## 📁 Estructura del Backend

```
backend/
├── manage.py                          # Comando de gestión de Django
├── requirements-django.txt            # Dependencias de Django
├── setup_django.sh                   # Script de configuración automática
├── env_example.txt                   # Ejemplo de configuración de entorno
├── quantitative_finance/             # Configuración principal del proyecto
│   ├── __init__.py
│   ├── settings.py                   # Configuración de Django
│   ├── urls.py                       # URLs principales
│   ├── wsgi.py                       # Configuración WSGI
│   └── asgi.py                       # Configuración ASGI
├── option_pricing/                   # Aplicación de pricing de opciones
│   ├── models.py                     # Modelos de datos
│   ├── serializers.py               # Serializers para API
│   ├── views.py                      # Vistas de API
│   ├── urls.py                       # URLs de la aplicación
│   ├── admin.py                      # Configuración del admin
│   └── services/                     # Servicios de cálculo
│       ├── black_scholes_service.py
│       ├── binomial_service.py
│       └── monte_carlo_service.py
└── portfolio_management/             # Aplicación de gestión de portfolios
    ├── models.py                     # Modelos para análisis
    ├── urls.py                       # URLs de la aplicación
    └── admin.py                      # Configuración del admin
```

## 🚀 Inicio Rápido

### 1. Configuración Automática
```bash
cd backend
./setup_django.sh
```

### 2. Configuración Manual
```bash
cd backend

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

## 🔗 URLs del Backend

- **API Option Pricing**: `http://127.0.0.1:8000/api/option-pricing/`
- **API Portfolio Management**: `http://127.0.0.1:8000/api/portfolio-management/`
- **Panel Admin**: `http://127.0.0.1:8000/admin/`
- **Health Check**: `http://127.0.0.1:8000/api/option-pricing/health/`

## 📊 Funcionalidades Implementadas

### Option Pricing
- ✅ Pricing con Black-Scholes, Binomial y Monte Carlo
- ✅ Cálculo de griegas (Delta, Gamma, Vega, Theta, Rho)
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

## 🔧 Configuración

### Variables de Entorno
Copia `env_example.txt` a `.env` y configura:
```bash
cp env_example.txt .env
# Edita .env con tus configuraciones
```

### Base de Datos
Por defecto usa SQLite. Para PostgreSQL:
```bash
# Instala psycopg2
pip install psycopg2-binary

# Configura DATABASE_URL en .env
```

## 🧪 Testing

```bash
# Ejecutar tests
python manage.py test

# Con coverage
pip install coverage
coverage run --source='.' manage.py test
coverage report
```

## 📚 Documentación

- **README Principal**: `../README_DJANGO.md`
- **Resumen de Conversión**: `../CONVERSION_SUMMARY.md`
- **Documentación de API**: Accesible en `/api/` cuando el servidor esté corriendo

## 🚀 Despliegue

### Desarrollo
```bash
python manage.py runserver
```

### Producción
```bash
# Usar gunicorn
pip install gunicorn
gunicorn quantitative_finance.wsgi:application

# O usar uvicorn para ASGI
pip install uvicorn
uvicorn quantitative_finance.asgi:application
```

## 🔍 Troubleshooting

### Problemas Comunes

1. **Error de migraciones**: `python manage.py makemigrations && python manage.py migrate`
2. **Dependencias faltantes**: `pip install -r requirements-django.txt`
3. **Permisos de script**: `chmod +x setup_django.sh`
4. **Base de datos corrupta**: Eliminar `db.sqlite3` y ejecutar migraciones

### Logs
Los logs se guardan en `logs/django.log` por defecto.

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

---

**¡El backend Django está listo para funcionar!** 🎉
