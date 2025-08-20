# Configuración del Frontend

Configuración simple y fácil de usar para el frontend.

## 📁 Estructura

```
config/
├── index.js      # Configuración principal
├── services.js   # Configuración de APIs
├── theme.js      # Configuración de estilos
└── README.md     # Este archivo
```

## 🚀 Uso Básico

### 1. Configuración Principal

```javascript
import config from '@/config';

// Valores básicos
const apiUrl = config.API_BASE_URL;
const isDev = config.isDevelopment();
const appName = config.APP_NAME;

// Helper para URLs de API
const userEndpoint = config.getApiUrl('/api/users/');
```

### 2. Configuración de Servicios

```javascript
import servicesConfig from '@/config/services';

// Backend API
const backendUrl = servicesConfig.backend.baseUrl;
const portfolioEndpoint = servicesConfig.backend.endpoints.portfolio;

// Yahoo Finance
const yahooUrl = servicesConfig.yahooFinance.baseUrl;
```

### 3. Configuración de Temas

```javascript
import themeConfig from '@/config/theme';

// Colores
const primaryColor = themeConfig.colors.primary[500];
const successColor = themeConfig.colors.success[500];

// Tipografía
const fontSize = themeConfig.typography.fontSize.lg;
```

## ⚙️ Variables de Entorno

Crea un archivo `.env.local` en la raíz del frontend:

```bash
# API
VITE_API_BASE_URL=http://localhost:8000

# App
VITE_NODE_ENV=development
VITE_APP_NAME=Quantitative Analysis Finance

# Features
VITE_ENABLE_DEBUG_MODE=true
VITE_THEME=light
VITE_LANGUAGE=es
```

## 🔧 Configuración Rápida

1. **Copiar variables de entorno**:
   ```bash
   cp env.example .env.local
   ```

2. **Editar `.env.local`** con tus valores

3. **Reiniciar el servidor** de desarrollo

## 📝 Notas

- Todas las variables deben comenzar con `VITE_`
- Nunca committear `.env.local` al repositorio
- La configuración se valida automáticamente
