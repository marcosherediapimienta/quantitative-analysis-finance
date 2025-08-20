# Troubleshooting: Pantalla en Blanco al Calcular Precio 🚨

## **Descripción del Problema**
La aplicación se queda con la pantalla en blanco cuando se intenta calcular el precio de una opción en el paso 3.

## **Causas Posibles**

### 1. **Error en la Validación de Datos**
- Valores inválidos en los campos de entrada
- Problemas con la función `validateAndFormatValue`
- Datos faltantes o corruptos

### 2. **Error en la Comunicación con el Backend**
- Endpoint no responde
- Error en la creación de la opción
- Error en el cálculo del precio
- Problemas de CORS o conectividad

### 3. **Error en el Estado de React**
- Estado corrupto o inconsistente
- Error en el renderizado del componente
- Problema con los hooks de React

### 4. **Error JavaScript No Manejado**
- Excepción que rompe la aplicación
- Promesa rechazada no manejada
- Error de sintaxis o runtime

## **Soluciones Paso a Paso**

### **Paso 1: Verificar Consola del Navegador**

1. **Abrir DevTools** (F12)
2. **Ir a la pestaña Console**
3. **Buscar errores** (líneas en rojo)
4. **Revisar logs** de la aplicación

**Logs esperados durante el cálculo:**
```
🧮 Calculando precio de la opción...
📊 Datos de la opción: {...}
🔍 Validando spot: 227.59 number
✅ Spot validado: 227.59
🔍 Validando strike: 225 number
✅ Strike validado: 225
...
```

### **Paso 2: Usar Botón de Debug**

1. **Hacer clic en "🐛 Debug"** en la interfaz
2. **Revisar la consola** para ver el estado actual
3. **Verificar que todos los campos** tengan valores válidos

### **Paso 3: Verificar Conectividad del Backend**

1. **Ir al paso 4 (Análisis de Sensibilidad)**
2. **Hacer clic en "🔍 Probar Conectividad del Backend"**
3. **Revisar logs** en la consola

### **Paso 4: Verificar Datos de Entrada**

**Campos obligatorios:**
- ✅ Símbolo de la acción
- ✅ Tipo de opción (CALL/PUT)
- ✅ Estilo de opción (Americana/Europea)
- ✅ Fecha de expiración
- ✅ Strike price
- ✅ Tasa libre de riesgo
- ✅ Volatilidad (o usar volatilidad implícita)

### **Paso 5: Verificar Validaciones**

**Límites de los campos:**
- **Volatilidad**: 0.000001 - 99.999999
- **Tasa de interés**: 0.000001 - 99.999999
- **Madurez**: 0.000001 - 99.999999 años
- **Spot/Strike**: 0.000001 - 999999999.999999

## **Problemas Comunes y Soluciones**

### **Problema 1: Valores Inválidos en Campos**

**Síntomas:**
- Error en consola: "Valor inválido para [campo]"
- Aplicación se rompe durante validación

**Solución:**
```javascript
// En la consola del navegador
console.log('User Inputs:', userInputs);
console.log('Yahoo Data:', yahooData);

// Verificar que los valores sean números válidos
Object.entries(userInputs).forEach(([key, value]) => {
  console.log(`${key}:`, value, typeof value, isNaN(value));
});
```

### **Problema 2: Error en Backend**

**Síntomas:**
- Error HTTP en consola
- Mensaje: "Error creando opción" o "Error calculando precio"

**Solución:**
1. **Verificar que el backend esté corriendo**
2. **Revisar logs del backend** en el terminal
3. **Probar endpoints manualmente** con curl

### **Problema 3: Estado Corrupto**

**Síntomas:**
- Variables undefined o null
- Objetos con estructura incorrecta

**Solución:**
1. **Hacer clic en "🐛 Debug"** para ver el estado
2. **Hacer clic en "🔄 Reiniciar"** para limpiar el estado
3. **Volver a empezar** desde el paso 1

### **Problema 4: Error JavaScript Crítico**

**Síntomas:**
- Pantalla completamente en blanco
- Error global capturado en la interfaz
- Aplicación no responde

**Solución:**
1. **Revisar error global** en la interfaz
2. **Hacer clic en "Cerrar"** para ocultar el error
3. **Hacer clic en "🔄 Reiniciar"** para limpiar todo
4. **Recargar la página** si persiste

## **Comandos de Debugging**

### **0. Archivo de Debug Automático**
```javascript
// Copiar y pegar este archivo en la consola del navegador:
// test-calculation-debug.js

// Luego ejecutar:
debugCalculation.runAllTests()
```

### **1. Verificar Estado Completo**
```javascript
// En la consola del navegador
console.log('=== ESTADO COMPLETO DE LA APLICACIÓN ===');
console.log('Current Step:', currentStep);
console.log('User Inputs:', userInputs);
console.log('Yahoo Data:', yahooData);
console.log('Analysis Results:', analysisResults);
console.log('Loading:', loading);
console.log('Error:', error);
console.log('Global Error:', globalError);
```

### **2. Verificar Validaciones**
```javascript
// Probar función de validación
const testValue = (value, fieldName) => {
  try {
    const result = validateAndFormatValue(value, fieldName);
    console.log(`✅ ${fieldName} válido:`, result);
    return true;
  } catch (error) {
    console.error(`❌ ${fieldName} inválido:`, error.message);
    return false;
  }
};

// Probar campos críticos
testValue(227.59, 'spot');
testValue(225, 'strike');
testValue(0.25, 'volatility');
testValue(0.05, 'rate');
```

### **3. Verificar Backend**
```javascript
// Probar endpoint de health
fetch('http://localhost:8000/api/option-pricing/health/')
  .then(response => response.json())
  .then(data => console.log('✅ Health check:', data))
  .catch(error => console.error('❌ Health check error:', error));

// Probar endpoint de opciones
fetch('http://localhost:8000/api/option-pricing/options/')
  .then(response => response.json())
  .then(data => console.log('✅ Options endpoint:', data))
  .catch(error => console.error('❌ Options error:', error));
```

## **Prevención de Problemas**

### **1. Validación de Entrada**
- **Siempre llenar todos los campos** antes de calcular
- **Verificar que los valores sean números válidos**
- **Usar rangos recomendados** para cada campo

### **2. Verificación de Backend**
- **Confirmar que el backend esté corriendo** antes de usar
- **Probar conectividad** regularmente
- **Revisar logs del backend** si hay problemas

### **3. Manejo de Errores**
- **No cerrar la consola** durante el uso
- **Revisar mensajes de error** completos
- **Usar el botón de debug** cuando sea necesario

## **Verificación Rápida**

### **Checklist de Diagnóstico:**
- [ ] Consola del navegador abierta (F12)
- [ ] Backend corriendo en puerto 8000
- [ ] Todos los campos obligatorios llenos
- [ ] Valores dentro de rangos válidos
- [ ] Botón de debug usado para verificar estado
- [ ] Conectividad del backend probada

### **Si el Problema Persiste:**
1. **Recargar la página** (Ctrl+F5)
2. **Limpiar caché del navegador**
3. **Verificar que no haya errores de JavaScript**
4. **Revisar la consola del backend**
5. **Contactar soporte** con logs completos

## **Logs de Error Comunes**

### **Error de Validación:**
```
❌ Error validando volatility: Error: Volatilidad excede el límite máximo de 99.999999
```

### **Error de Backend:**
```
❌ Error creando opción: HTTP 400 - {"volatility":["Asegúrese de que no haya más de 8 dígitos en total."]}
```

### **Error de Estado:**
```
❌ Valor inválido para spot: undefined (tipo: undefined)
```

### **Error de Conectividad:**
```
❌ Error calculando precio: HTTP 404
```

---

*Esta guía es parte de GalaAnalytics Pro - Plataforma avanzada de análisis cuantitativo de opciones financieras.*

**Recuerda**: La mayoría de problemas de pantalla en blanco se resuelven revisando la consola del navegador y verificando la conectividad del backend. Usa las herramientas de debug integradas para diagnosticar problemas rápidamente.
