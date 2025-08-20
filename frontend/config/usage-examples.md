# Ejemplos de Uso

## 📱 En Componentes React

```jsx
import React from 'react';
import config from '@/config';
import servicesConfig from '@/config/services';
import themeConfig from '@/config/theme';

function MyComponent() {
  const handleApiCall = async () => {
    const url = config.getApiUrl('/api/users/');
    const response = await fetch(url);
    return response.json();
  };

  return (
    <div style={{ 
      backgroundColor: themeConfig.colors.primary[500],
      padding: themeConfig.spacing.md 
    }}>
      <h1 style={{ 
        fontSize: themeConfig.typography.fontSize.xl,
        color: 'white' 
      }}>
        {config.APP_NAME}
      </h1>
      
      {config.isDevelopment() && (
        <p>Modo desarrollo activo</p>
      )}
    </div>
  );
}
```

## 🔌 En Funciones de API

```javascript
import config from '@/config';
import servicesConfig from '@/config/services';

export const apiService = {
  // Llamada al backend
  async getPortfolio() {
    const url = config.getApiUrl('/api/portfolio/');
    const response = await fetch(url);
    return response.json();
  },

  // Llamada a Yahoo Finance
  async getStockPrice(symbol) {
    const endpoint = servicesConfig.yahooFinance.endpoints.quote
      .replace('{symbol}', symbol);
    const url = `${servicesConfig.yahooFinance.baseUrl}${endpoint}`;
    
    const response = await fetch(url);
    return response.json();
  }
};
```

## 🎨 En Estilos CSS-in-JS

```javascript
import themeConfig from '@/config/theme';

export const styles = {
  button: {
    backgroundColor: themeConfig.colors.primary[500],
    color: 'white',
    padding: themeConfig.spacing.md,
    borderRadius: themeConfig.borders.radius.md,
    fontSize: themeConfig.typography.fontSize.base,
    fontWeight: themeConfig.typography.fontWeight.medium,
    boxShadow: themeConfig.shadows.md
  },
  
  container: {
    padding: themeConfig.spacing.lg,
    maxWidth: themeConfig.breakpoints.lg
  }
};
```

## 🚀 Inicio Rápido

1. **Copiar variables de entorno**:
   ```bash
   cp env.example .env.local
   ```

2. **Importar en tu componente**:
   ```javascript
   import config from '@/config';
   ```

3. **Usar la configuración**:
   ```javascript
   const apiUrl = config.API_BASE_URL;
   const isDev = config.isDevelopment();
   ```
