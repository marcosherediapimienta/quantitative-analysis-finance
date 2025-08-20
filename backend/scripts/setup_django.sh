#!/bin/bash

# Script para configurar el proyecto Django
echo "🚀 Configurando proyecto Django para Quantitative Finance..."

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📚 Instalando dependencias de Django..."
pip install --upgrade pip
pip install -r requirements.txt

# Crear migraciones
echo "🗄️ Creando migraciones..."
python manage.py makemigrations option_pricing
python manage.py makemigrations portfolio_management

# Aplicar migraciones
echo "⚡ Aplicando migraciones a la base de datos..."
python manage.py migrate

# Crear superusuario (opcional)
echo "👤 ¿Deseas crear un superusuario? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    python manage.py createsuperuser
fi

# Recopilar archivos estáticos
echo "📁 Recopilando archivos estáticos..."
python manage.py collectstatic --noinput

echo "✅ ¡Configuración completada!"
echo ""
echo "🎯 Para ejecutar el servidor de desarrollo:"
echo "   python manage.py runserver"
echo ""
echo "🔗 URLs disponibles:"
echo "   - Admin: http://127.0.0.1:8000/admin/"
echo "   - API Option Pricing: http://127.0.0.1:8000/api/option-pricing/"
echo "   - API Portfolio Management: http://127.0.0.1:8000/api/portfolio-management/"
echo "   - Health Check: http://127.0.0.1:8000/api/option-pricing/health/"
