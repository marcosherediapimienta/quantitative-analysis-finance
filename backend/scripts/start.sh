#!/bin/bash

# Script de inicio para el backend Django
echo "🚀 Iniciando Backend Django - Quantitative Finance..."

# Verificar si estamos en el directorio correcto
if [ ! -f "manage.py" ]; then
    echo "❌ Error: No se encontró manage.py. Ejecuta este script desde el directorio backend/"
    exit 1
fi

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Verificar si las dependencias están instaladas
if ! python -c "import django" 2>/dev/null; then
    echo "📚 Instalando dependencias..."
    pip install -r requirements.txt
fi

# Verificar si la base de datos está configurada
if [ ! -f "db.sqlite3" ]; then
    echo "🗄️ Configurando base de datos..."
    python manage.py makemigrations
    python manage.py migrate
    
    echo "👤 Creando superusuario..."
    echo "Por favor, sigue las instrucciones para crear tu cuenta de administrador:"
    python manage.py createsuperuser
fi

# Recopilar archivos estáticos
echo "📁 Recopilando archivos estáticos..."
python manage.py collectstatic --noinput

# Iniciar servidor
echo "🌐 Iniciando servidor Django..."
echo "📍 URLs disponibles:"
echo "   - API: http://127.0.0.1:8000/api/"
echo "   - Admin: http://127.0.0.1:8000/admin/"
echo "   - Health Check: http://127.0.0.1:8000/api/option-pricing/health/"
echo ""
echo "🛑 Presiona Ctrl+C para detener el servidor"
echo ""

python manage.py runserver 0.0.0.0:8000
