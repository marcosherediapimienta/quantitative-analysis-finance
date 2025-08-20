// Configuración de temas y estilos
// Configuración simplificada y esencial

export const themeConfig = {
  // Colores principales
  colors: {
    primary: {
      50: '#eff6ff',
      100: '#dbeafe',
      500: '#3b82f6',
      600: '#2563eb',
      700: '#1d4ed8'
    },
    secondary: {
      50: '#f8fafc',
      100: '#f1f5f9',
      500: '#64748b',
      600: '#475569'
    },
    success: {
      500: '#22c55e',
      600: '#16a34a'
    },
    warning: {
      500: '#f59e0b',
      600: '#d97706'
    },
    error: {
      500: '#ef4444',
      600: '#dc2626'
    },
    // Colores específicos para finanzas
    finance: {
      bullish: '#10b981',
      bearish: '#ef4444',
      neutral: '#6b7280'
    }
  },
  
  // Tipografía
  typography: {
    fontFamily: {
      sans: ['Inter', 'system-ui', 'sans-serif'],
      mono: ['JetBrains Mono', 'monospace']
    },
    fontSize: {
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem'
    },
    fontWeight: {
      normal: '400',
      medium: '500',
      semibold: '600',
      bold: '700'
    }
  },
  
  // Espaciado
  spacing: {
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem'
  },
  
  // Bordes y sombras
  borders: {
    radius: {
      sm: '0.125rem',
      md: '0.375rem',
      lg: '0.5rem'
    }
  },
  
  shadows: {
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)'
  },
  
  // Breakpoints
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px'
  }
};

export default themeConfig;
