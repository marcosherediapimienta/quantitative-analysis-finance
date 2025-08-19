#!/usr/bin/env python
"""
Script de ejecución para el backend Django.
Permite ejecutar el servidor desde el directorio backend.
"""

import os
import sys
import django
from django.core.management import execute_from_command_line

def main():
    """Función principal para ejecutar comandos Django"""
    # Configurar el entorno Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'quantitative_finance.settings')
    
    # Configurar Django
    django.setup()
    
    # Ejecutar el comando Django
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
