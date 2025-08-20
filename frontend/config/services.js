// Configuración de servicios y APIs
// Configuración simplificada y esencial

export const servicesConfig = {
  // Backend API
  backend: {
    baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    endpoints: {
      // Option Pricing
      binomialPricing: '/api/option-pricing/binomial/',
      blackScholesPricing: '/api/option-pricing/black-scholes/',
      monteCarloPricing: '/api/option-pricing/monte-carlo/',
      
      // Portfolio Management
      portfolio: '/api/portfolio/',
      portfolioHistory: '/api/portfolio/history/',
      
      // User Management
      userProfile: '/api/user/profile/',
      
      // Authentication
      login: '/api/auth/login/',
      logout: '/api/auth/logout/',
      register: '/api/auth/register/'
    },
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    timeout: 10000
  },
  
  // Yahoo Finance API
  yahooFinance: {
    baseUrl: 'https://query1.finance.yahoo.com',
    endpoints: {
      quote: '/v8/finance/chart/{symbol}',
      search: '/v1/finance/search',
      history: '/v8/finance/chart/{symbol}?interval={interval}&range={range}'
    }
  },
  
  // Alpha Vantage API
  alphaVantage: {
    baseUrl: 'https://www.alphavantage.co/query',
    endpoints: {
      quote: '?function=GLOBAL_QUOTE&symbol={symbol}&apikey={apiKey}',
      timeSeries: '?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={apiKey}'
    }
  }
};

export default servicesConfig;
