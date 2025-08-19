#!/bin/bash

# Script de instalación para el backend Django
echo "🔧 Instalando Backend Django - Quantitative Finance..."

# Verificar si estamos en el directorio correcto
if [ ! -f "manage.py" ]; then
    echo "❌ Error: No se encontró manage.py. Ejecuta este script desde el directorio backend/"
    exit 1
fi

# Verificar si Python 3 está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado. Por favor, instala Python 3.9+"
    exit 1
fi

# Verificar si pip está instalado
if ! command -v pip &> /dev/null; then
    echo "❌ Error: pip no está instalado. Por favor, instala pip"
    exit 1
fi

echo "✅ Python 3 y pip están instalados"

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
if [ -d "venv" ]; then
    echo "⚠️  El entorno virtual ya existe. ¿Deseas recrearlo? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "📚 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Verificar instalación de Django
if ! python -c "import django; print('Django version:', django.get_version())" 2>/dev/null; then
    echo "❌ Error: Django no se instaló correctamente"
    exit 1
fi

echo "✅ Django instalado correctamente"

# Crear directorio de logs
echo "📁 Creando directorio de logs..."
mkdir -p logs

# Configurar base de datos
echo "🗄️ Configurando base de datos..."
python manage.py makemigrations option_pricing
python manage.py makemigrations portfolio_management
python manage.py migrate

echo "✅ Base de datos configurada"

# Recopilar archivos estáticos
echo "📁 Recopilando archivos estáticos..."
python manage.py collectstatic --noinput

echo ""
echo "🎉 ¡Instalación completada exitosamente!"
echo ""
echo "📋 Próximos pasos:"
echo "1. Crear superusuario: python manage.py createsuperuser"
echo "2. Iniciar servidor: ./start.sh"
echo "3. Acceder a la API: http://127.0.0.1:8000/api/"
echo "4. Acceder al admin: http://127.0.0.1:8000/admin/"
echo ""
echo "🚀 Para iniciar el servidor, ejecuta: ./start.sh"
