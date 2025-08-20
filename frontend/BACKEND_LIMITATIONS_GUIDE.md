# Guía de Limitaciones del Backend ⚠️

## Problemas Comunes y Soluciones

### 1. Error de Volatilidad: "más de 8 dígitos en total"

#### **Descripción del Error**
```
"volatility":["Asegúrese de que no haya más de 8 dígitos en total."]
```

#### **Causa**
El modelo del backend tiene restricciones estrictas en los campos decimales:
- **Volatilidad**: `max_digits=8, decimal_places=6`
- **Tasa de Interés**: `max_digits=8, decimal_places=6`
- **Madurez**: `max_digits=8, decimal_places=6`

Esto significa que el valor máximo para estos campos es **99.999999**.

#### **Soluciones Implementadas**

##### **Frontend (Validación Automática)**
- ✅ Validación en tiempo real de rangos
- ✅ Formateo automático a 6 decimales
- ✅ Mensajes de error preventivos
- ✅ Límites en inputs (min/max)

##### **Backend (Validación del Modelo)**
- ✅ Restricción de `max_digits=8`
- ✅ Restricción de `decimal_places=6`
- ✅ Validadores de Django

#### **Rangos Válidos por Campo**

| Campo | Rango Mínimo | Rango Máximo | Ejemplo Válido |
|-------|--------------|---------------|----------------|
| **Volatilidad** | 0.000001 | 99.999999 | 25.500000 |
| **Tasa de Interés** | 0.000001 | 99.999999 | 5.250000 |
| **Madurez** | 0.000001 | 99.999999 | 0.250000 |

### 2. Campos de Precio (Spot/Strike)

#### **Restricciones**
- **Spot**: `max_digits=15, decimal_places=6`
- **Strike**: `max_digits=15, decimal_places=6`

#### **Rango Válido**
- **Mínimo**: 0.000001
- **Máximo**: 999,999,999.999999

#### **Ejemplos Válidos**
- ✅ $150.50
- ✅ $1,250.75
- ✅ $0.01
- ❌ $1,000,000,000.00 (excede límite)

### 3. Campos de Griegas

#### **Restricciones**
- **Delta, Gamma, Vega, Theta, Rho**: `max_digits=15, decimal_places=8`

#### **Rango Válido**
- **Mínimo**: -999,999,999.99999999
- **Máximo**: 999,999,999.99999999

## 🔧 Soluciones Técnicas Implementadas

### 1. **Función de Validación Automática**

```javascript
const validateAndFormatValue = (value, fieldName) => {
  switch (fieldName) {
    case 'volatility':
      if (value > 99.999999) {
        throw new Error(`Volatilidad excede el límite máximo de 99.999999`);
      }
      return parseFloat(value.toFixed(6));
    
    case 'rate':
      if (value > 99.999999) {
        throw new Error(`Tasa de interés excede el límite máximo de 99.999999`);
      }
      return parseFloat(value.toFixed(6));
    
    // ... otros campos
  }
};
```

### 2. **Formateo Automático**

```javascript
// Antes (problemático)
volatility: userInputs.useImpliedVolatility ? 
  (yahooData.selectedOption.implied_volatility || 0.25) : 
  (parseFloat(userInputs.volatility) / 100)

// Después (seguro)
volatility: validateAndFormatValue(
  userInputs.useImpliedVolatility ? 
    (yahooData.selectedOption.implied_volatility || 0.25) : 
    (parseFloat(userInputs.volatility) / 100),
  'volatility'
)
```

### 3. **Validación en Tiempo Real**

```javascript
// Input con validación
<Input
  type="number"
  step="0.1"
  min="0.1"
  max="99.9"
  value={userInputs.volatility}
  onChange={(e) => updateUserInput('volatility', e.target.value)}
/>

// Mensaje de validación
{userInputs.volatility && parseFloat(userInputs.volatility) > 99.9 && (
  <div className="text-xs text-red-400">
    ⚠️ La volatilidad excede el límite máximo del backend
  </div>
)}
```

## 📊 Casos de Uso Comunes

### 1. **Volatilidad Implícita de Yahoo Finance**

#### **Problema**
Los valores de volatilidad implícita pueden ser muy pequeños (ej: 0.285 para 28.5%)

#### **Solución**
- ✅ Formateo automático a 6 decimales
- ✅ Validación de límites antes del envío
- ✅ Conversión segura de porcentajes

#### **Ejemplo**
```javascript
// Valor de Yahoo Finance
implied_volatility: 0.285

// Formateado para backend
volatility: 0.285000

// Validado
if (0.285000 > 99.999999) // false ✅
```

### 2. **Tasas de Interés**

#### **Problema**
Las tasas pueden ser muy pequeñas (ej: 0.0525 para 5.25%)

#### **Solución**
- ✅ Conversión de porcentaje a decimal
- ✅ Formateo a 6 decimales
- ✅ Validación de rangos

#### **Ejemplo**
```javascript
// Input del usuario
riskFreeRate: "5.25"

// Conversión
rate: 5.25 / 100 = 0.0525

// Formateado
rate: 0.052500

// Validado
if (0.052500 > 99.999999) // false ✅
```

### 3. **Madurez (Tiempo al Vencimiento)**

#### **Problema**
La madurez se calcula en días y se convierte a años

#### **Solución**
- ✅ Cálculo preciso de días
- ✅ Conversión a años con 6 decimales
- ✅ Validación de límites

#### **Ejemplo**
```javascript
// Días hasta expiración
days: 30

// Conversión a años
maturity: 30 / 365 = 0.0821917808219178

// Formateado
maturity: 0.082192

// Validado
if (0.082192 > 99.999999) // false ✅
```

## 🚀 Mejores Prácticas

### 1. **Validación Preventiva**
- ✅ Validar rangos antes de enviar al backend
- ✅ Formatear números a la precisión correcta
- ✅ Mostrar mensajes de error claros

### 2. **Manejo de Errores**
- ✅ Capturar errores del backend
- ✅ Mostrar mensajes de error específicos
- ✅ Sugerir soluciones al usuario

### 3. **Documentación**
- ✅ Explicar limitaciones claramente
- ✅ Proporcionar ejemplos de valores válidos
- ✅ Guiar al usuario en la configuración

## 🔍 Debugging

### 1. **Verificar Valores Antes del Envío**

```javascript
console.log('Valores a enviar:', {
  spot: spot,
  strike: strike,
  maturity: maturity,
  volatility: volatility,
  rate: rate
});
```

### 2. **Verificar Respuesta del Backend**

```javascript
if (!createResponse.ok) {
  const errorData = await createResponse.json();
  console.error('❌ Error response:', errorData);
  throw new Error(`Error creando opción: HTTP ${createResponse.status} - ${JSON.stringify(errorData)}`);
}
```

### 3. **Validar Formato de Números**

```javascript
// Verificar que no exceda límites
console.log('Volatilidad:', volatility, 'Límite:', 99.999999);
console.log('Tasa:', rate, 'Límite:', 99.999999);
console.log('Madurez:', maturity, 'Límite:', 99.999999);
```

## 📚 Referencias

### **Modelos del Backend**
- `Option.volatility`: DecimalField(max_digits=8, decimal_places=6)
- `Option.rate`: DecimalField(max_digits=8, decimal_places=6)
- `Option.maturity`: DecimalField(max_digits=8, decimal_places=6)
- `Option.spot`: DecimalField(max_digits=15, decimal_places=6)
- `Option.strike`: DecimalField(max_digits=15, decimal_places=6)

### **Validadores Django**
- `MinValueValidator(0.000001)` para campos positivos
- `MinValueValidator(0)` para campos no negativos

### **Documentación Relacionada**
- [Guía de Opciones Americanas](./AMERICAN_OPTIONS_GUIDE.md)
- [Guía de Análisis de Sensibilidad](./SENSITIVITY_ANALYSIS_GUIDE.md)

---

*Esta guía es parte de GalaAnalytics Pro - Plataforma avanzada de análisis cuantitativo de opciones financieras.*

**Recuerda**: Las limitaciones del backend están diseñadas para garantizar la precisión y estabilidad de los cálculos. Siempre valida los datos antes de enviarlos y utiliza las funciones de formateo automático implementadas.
