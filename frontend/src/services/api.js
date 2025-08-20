// Servicio de API para conectar con el backend
import config from '@/config';

class ApiService {
  constructor() {
    this.baseURL = config.API_BASE_URL;
    this.timeout = config.API_TIMEOUT;
  }

  // Método para hacer peticiones HTTP
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultOptions = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Configurar timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    
    try {
      const response = await fetch(url, {
        ...defaultOptions,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (config.isDevelopment()) {
        console.error('Error en API call:', error);
      }
      
      throw error;
    }
  }

  // Métodos para Option Pricing
  async getOptionPricingHealth() {
    return this.request('/api/option-pricing/health/');
  }

  async getYahooFinanceTest() {
    return this.request('/api/option-pricing/yahoo-finance/test/');
  }

  async getYahooFinanceTicker(symbol) {
    return this.request(`/api/option-pricing/yahoo-finance/ticker/${symbol}/`);
  }

  async getOptions() {
    return this.request('/api/option-pricing/options/');
  }

  async getOption(id) {
    return this.request(`/api/option-pricing/options/${id}/`);
  }

  async createOption(data) {
    return this.request('/api/option-pricing/options/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateOption(id, data) {
    return this.request(`/api/option-pricing/options/${id}/`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteOption(id) {
    return this.request(`/api/option-pricing/options/${id}/`, {
      method: 'DELETE',
    });
  }

  // Métodos para Portfolio Management
  async getPortfolioManagementHealth() {
    return this.request('/api/portfolio-management/health/');
  }

  // Métodos para Portfolios
  async getPortfolios() {
    return this.request('/api/option-pricing/portfolios/');
  }

  async getPortfolio(id) {
    return this.request(`/api/option-pricing/portfolios/${id}/`);
  }

  async createPortfolio(data) {
    return this.request('/api/option-pricing/portfolios/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updatePortfolio(id, data) {
    return this.request(`/api/option-pricing/portfolios/${id}/`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deletePortfolio(id) {
    return this.request(`/api/option-pricing/portfolios/${id}/`, {
      method: 'DELETE',
    });
  }

  // Métodos para Pricing Results
  async getPricingResults() {
    return this.request('/api/option-pricing/pricing-results/');
  }

  async getPricingResult(id) {
    return this.request(`/api/option-pricing/pricing-results/${id}/`);
  }

  async createPricingResult(data) {
    return this.request('/api/option-pricing/pricing-results/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Método para calcular precios de opciones
  async calculateOptionPrice(optionData) {
    // Aquí puedes implementar la lógica para diferentes modelos de pricing
    // Por ahora, creamos un pricing result
    return this.createPricingResult({
      option: optionData,
      model: 'black_scholes', // o 'binomial', 'monte_carlo'
      calculated_price: null, // Se calculará en el backend
      timestamp: new Date().toISOString(),
    });
  }

  // Método para obtener datos de mercado
  async getMarketData(symbol) {
    return this.getYahooFinanceTicker(symbol);
  }

  // Método para verificar la salud de todos los servicios
  async checkAllServices() {
    try {
      const [optionPricing, portfolioManagement] = await Promise.all([
        this.getOptionPricingHealth(),
        this.getPortfolioManagementHealth(),
      ]);

      return {
        optionPricing,
        portfolioManagement,
        allServices: 'healthy',
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        allServices: 'unhealthy',
        error: error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }
}

// Crear una instancia del servicio
const apiService = new ApiService();

export default apiService;
