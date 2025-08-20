// Configuración principal del frontend
// Configuración simplificada y fácil de usar

const config = {
  // API Configuration
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  API_TIMEOUT: parseInt(import.meta.env.VITE_API_TIMEOUT) || 10000,
  
  // Environment
  NODE_ENV: import.meta.env.VITE_NODE_ENV || 'development',
  APP_NAME: import.meta.env.VITE_APP_NAME || 'Quantitative Analysis Finance',
  
  // Feature Flags
  ENABLE_DEBUG_MODE: import.meta.env.VITE_ENABLE_DEBUG_MODE === 'true' || true,
  ENABLE_ANALYTICS: import.meta.env.VITE_ENABLE_ANALYTICS === 'true' || false,
  
  // UI Configuration
  THEME: import.meta.env.VITE_THEME || 'light',
  LANGUAGE: import.meta.env.VITE_LANGUAGE || 'es',
  
  // Helper methods
  isDevelopment: () => config.NODE_ENV === 'development',
  isProduction: () => config.NODE_ENV === 'production',
  
  // API helpers
  getApiUrl: (endpoint) => {
    const baseUrl = config.API_BASE_URL;
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    return `${baseUrl}${cleanEndpoint}`;
  }
};

export default config;
