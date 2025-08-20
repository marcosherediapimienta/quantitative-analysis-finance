
# Frontend React - Quantitative Analysis Finance

Este proyecto es un frontend creado con React y Vite. Está diseñado para integrarse con el backend de análisis cuantitativo financiero.

## 🚀 Funcionalidades Principales

### Análisis de Opciones
- **Opciones Europeas**: Análisis tradicional con Black-Scholes, Binomial y Monte Carlo
- **Opciones Americanas**: Análisis avanzado con ejercicio anticipado
- **Integración con Yahoo Finance**: Datos en tiempo real de opciones
- **Cálculo de Griegas**: Delta, Gamma, Vega, Theta, Rho
- **Análisis de Sensibilidad**: Análisis completo de sensibilidad a parámetros del mercado
  - Sensibilidad al precio del subyacente (Spot)
  - Sensibilidad al precio de ejercicio (Strike)
  - Sensibilidad a la volatilidad
  - Sensibilidad a la tasa de interés
  - Sensibilidad al tiempo (Theta decay)

### Modelos de Pricing
- **Black-Scholes**: Modelo analítico clásico (solo opciones europeas)
- **Binomial**: Árbol binomial para opciones americanas y europeas
- **Monte Carlo**: Simulación estocástica para casos complejos

### Características Especiales
- **Flujo Interactivo**: Análisis paso a paso guiado
- **Validación Inteligente**: Prevención de errores de configuración
- **Interfaz Responsiva**: Diseño moderno y adaptativo
- **Análisis Comparativo**: Comparación entre estilos de opciones

## 📚 Documentación

- [Guía de Opciones Americanas](./AMERICAN_OPTIONS_GUIDE.md) - Guía completa sobre opciones americanas
- [Guía de Análisis de Sensibilidad](./SENSITIVITY_ANALYSIS_GUIDE.md) - Guía completa sobre análisis de sensibilidad
- [Guía de Limitaciones del Backend](./BACKEND_LIMITATIONS_GUIDE.md) - Soluciones a errores comunes y limitaciones
- [Troubleshooting de Sensibilidad](./SENSITIVITY_TROUBLESHOOTING.md) - Solución a problemas del análisis de sensibilidad
- [Troubleshooting de Pantalla en Blanco](./WHITE_SCREEN_TROUBLESHOOTING.md) - Solución a problemas de pantalla en blanco
- [Ejemplos de Uso](./config/usage-examples.md) - Casos de uso y ejemplos prácticos

## 🛠️ Scripts principales

- `npm run dev`: Inicia el servidor de desarrollo.
- `npm run build`: Compila la aplicación para producción.
- `npm run preview`: Previsualiza la build de producción.

## 🏗️ Estructura

- `/src`: Código fuente principal
  - `/components`: Componentes reutilizables
  - `/hooks`: Hooks personalizados de React
  - `/pages`: Páginas principales de la aplicación
  - `/services`: Servicios de API y lógica de negocio
- `/public`: Archivos estáticos
- `/config`: Configuración y ejemplos

## 🔌 Integración

Asegúrate de tener el backend corriendo y actualiza los endpoints en los servicios del frontend según sea necesario.

### Requisitos del Backend
- Django con Django REST Framework
- Servicios de pricing de opciones (Black-Scholes, Binomial, Monte Carlo)
- API de Yahoo Finance para datos de mercado
- Soporte para opciones americanas y europeas

## 🎯 Casos de Uso

### 1. Análisis de Opciones Americanas
- Selección de estilo de opción (americana/europea)
- Validación automática de modelos compatibles
- Cálculo preciso considerando ejercicio anticipado
- Análisis de ventajas y consideraciones

### 2. Análisis de Sensibilidad
- Análisis completo de sensibilidad a parámetros del mercado
- Visualización gráfica de resultados
- Configuración flexible de rangos y precisión
- Interpretación detallada de cambios en precios

### 3. Comparación de Modelos
- Evaluación de precisión vs. velocidad
- Selección automática del modelo óptimo
- Parámetros específicos por modelo
- Resultados comparativos

### 4. Gestión de Portfolio
- Análisis de posiciones existentes
- Evaluación de nuevas oportunidades
- Gestión de riesgo con griegas
- Optimización de estrategias

## 🔧 Configuración

### Variables de Entorno
```bash
# Ejemplo de configuración
VITE_API_BASE_URL=http://localhost:8000
VITE_YAHOO_FINANCE_ENABLED=true
```

### Dependencias Principales
- React 18+
- Framer Motion (animaciones)
- Lucide React (iconos)
- Tailwind CSS (estilos)

## 📱 Características de UX

- **Navegación Intuitiva**: Flujo paso a paso guiado
- **Validación en Tiempo Real**: Prevención de errores
- **Feedback Visual**: Indicadores de estado y progreso
- **Responsive Design**: Optimizado para todos los dispositivos
- **Temas Oscuros**: Interfaz moderna y elegante

## 🚧 Estado del Proyecto

- ✅ Análisis de opciones europeas
- ✅ Análisis de opciones americanas
- ✅ Integración con Yahoo Finance
- ✅ Modelos de pricing avanzados
- ✅ Cálculo de griegas
- ✅ Análisis de sensibilidad completo
- 🔄 Portfolio management (en desarrollo)
- 🔄 Configuración avanzada (en desarrollo)

---

Para más detalles sobre la integración, consulta la documentación del backend y la [Guía de Opciones Americanas](./AMERICAN_OPTIONS_GUIDE.md).

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
