# Troubleshooting del Análisis de Sensibilidad 🔧

## Error HTTP 404: Endpoint No Encontrado

### **Descripción del Error**
```
Error en análisis de sensibilidad: HTTP 404
```

### **Causas Posibles**

#### 1. **Backend No Está Corriendo**
- El servidor Django no está ejecutándose
- El puerto 8000 no está disponible
- Hay un error en el backend

#### 2. **Endpoint No Está Configurado**
- La acción `sensitivity_analysis` no está registrada
- Problema con las URLs del router
- Error en la configuración del ViewSet

#### 3. **Problema de Conectividad**
- Firewall bloqueando la conexión
- CORS no configurado correctamente
- Problema de red

### **Soluciones Paso a Paso**

#### **Paso 1: Verificar Estado del Backend**

1. **Abrir la consola del navegador** (F12 → Console)
2. **Hacer clic en "🔍 Probar Conectividad del Backend"**
3. **Revisar los logs** en la consola

#### **Paso 2: Verificar Endpoints**

Los logs deberían mostrar:
```
✅ Health check: 200 true
✅ Options endpoint: 200 true
✅ Sensitivity endpoint test: 400 false (esperado con ID inválido)
```

Si ves `404` en el endpoint de sensibilidad, hay un problema de configuración.

#### **Paso 3: Verificar Backend**

1. **Ir al terminal donde corre el backend**
2. **Verificar que no hay errores**
3. **Confirmar que el servidor está en puerto 8000**

```bash
# En el terminal del backend
python manage.py runserver 8000
```

#### **Paso 4: Verificar URLs del Backend**

En `backend/apps/option_pricing/urls.py` debe estar:

```python
router = DefaultRouter()
router.register(r'options', OptionViewSet)
```

#### **Paso 5: Verificar ViewSet**

En `backend/apps/option_pricing/views.py` debe estar:

```python
@action(detail=True, methods=['post'])
def sensitivity_analysis(self, request, pk=None):
    # ... implementación
```

### **Verificación Manual de Endpoints**

#### **1. Health Check**
```bash
curl http://localhost:8000/api/option-pricing/health/
```

**Respuesta esperada:**
```json
{"status": "ok", "message": "Backend funcionando correctamente"}
```

#### **2. Endpoint de Opciones**
```bash
curl http://localhost:8000/api/option-pricing/options/
```

**Respuesta esperada:**
```json
[]
```

#### **3. Endpoint de Sensibilidad (con ID inválido)**
```bash
curl -X POST http://localhost:8000/api/option-pricing/options/999999/sensitivity_analysis/ \
  -H "Content-Type: application/json" \
  -d '{"sensitivity_type": "spot", "min_value": 100, "max_value": 200, "num_points": 5, "model": "binomial"}'
```

**Respuesta esperada:**
```json
{"detail": "No encontrado."}
```

### **Problemas Comunes y Soluciones**

#### **Problema 1: Backend No Responde**

**Síntomas:**
- Error de conexión en la consola
- Timeout en las peticiones
- Health check falla

**Solución:**
```bash
# 1. Ir al directorio del backend
cd backend

# 2. Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# 3. Verificar dependencias
pip install -r requirements.txt

# 4. Ejecutar migraciones
python manage.py migrate

# 5. Iniciar servidor
python manage.py runserver 8000
```

#### **Problema 2: Endpoint 404**

**Síntomas:**
- Health check funciona
- Options endpoint funciona
- Solo sensitivity_analysis falla

**Solución:**
1. **Verificar que el ViewSet esté importado**
2. **Verificar que la acción esté decorada con @action**
3. **Reiniciar el servidor Django**

#### **Problema 3: Error de CORS**

**Síntomas:**
- Error de CORS en la consola
- Peticiones bloqueadas por el navegador

**Solución:**
En `backend/config/settings.py` debe estar:

```python
INSTALLED_APPS = [
    # ...
    'corsheaders',
    # ...
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    # ... otros middleware
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
]
```

### **Debugging Avanzado**

#### **1. Logs del Backend**

En `backend/config/settings.py`:

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}
```

#### **2. Verificar Base de Datos**

```bash
python manage.py shell
```

```python
from apps.option_pricing.models import Option
from apps.option_pricing.views import OptionViewSet

# Verificar que el modelo existe
print(Option.objects.count())

# Verificar que el ViewSet funciona
viewset = OptionViewSet()
print(dir(viewset))
```

#### **3. Verificar URLs Generadas**

```bash
python manage.py show_urls
```

Buscar:
```
/api/option-pricing/options/{id}/sensitivity_analysis/
```

### **Verificación del Frontend**

#### **1. Console Logs**

Verificar en la consola del navegador:
- ✅ Peticiones HTTP exitosas
- ✅ Respuestas del backend
- ❌ Errores de red o CORS

#### **2. Network Tab**

En DevTools → Network:
- Verificar que las peticiones se envían
- Verificar códigos de respuesta
- Verificar payloads enviados

#### **3. Estado de la Aplicación**

Verificar en React DevTools:
- Estado de `sensitivityParams`
- Estado de `sensitivityResults`
- Valores de `optionId`

### **Comandos de Verificación Rápida**

#### **Backend**
```bash
# Verificar estado
curl http://localhost:8000/api/option-pricing/health/

# Verificar opciones
curl http://localhost:8000/api/option-pricing/options/

# Verificar sensibilidad (debe dar 404 con ID inválido)
curl -X POST http://localhost:8000/api/option-pricing/options/999999/sensitivity_analysis/ \
  -H "Content-Type: application/json" \
  -d '{"sensitivity_type": "spot", "min_value": 100, "max_value": 200, "num_points": 5, "model": "binomial"}'
```

#### **Frontend**
```javascript
// En la consola del navegador
console.log('API Base URL:', 'http://localhost:8000');
console.log('Sensitivity Params:', sensitivityParams);
console.log('Option ID:', optionId);
```

### **Contacto y Soporte**

Si el problema persiste después de seguir esta guía:

1. **Revisar logs del backend** en el terminal
2. **Revisar console del navegador** para errores
3. **Verificar que todos los endpoints funcionan** manualmente
4. **Documentar el error específico** con logs completos

---

*Esta guía es parte de GalaAnalytics Pro - Plataforma avanzada de análisis cuantitativo de opciones financieras.*

**Recuerda**: La mayoría de problemas de conectividad se resuelven verificando el estado del backend y la configuración de URLs. Usa el botón de prueba de conectividad para diagnosticar problemas rápidamente.
