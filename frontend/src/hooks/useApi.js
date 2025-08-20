// Hook personalizado para usar la API del backend
import { useState, useEffect, useCallback } from 'react';
import apiService from '@/services/api';

export const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  // Función para hacer peticiones con manejo de estado
  const makeRequest = useCallback(async (apiCall, ...args) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await apiCall(...args);
      setData(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Función para limpiar el estado
  const clearState = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return {
    loading,
    error,
    data,
    makeRequest,
    clearState,
  };
};

// Hook específico para Option Pricing
export const useOptionPricing = () => {
  const { loading, error, data, makeRequest, clearState } = useApi();

  const getHealth = useCallback(() => {
    return makeRequest(apiService.getOptionPricingHealth);
  }, [makeRequest]);

  const getYahooFinanceTest = useCallback(() => {
    return makeRequest(apiService.getYahooFinanceTest);
  }, [makeRequest]);

  const getYahooFinanceTicker = useCallback((symbol) => {
    return makeRequest(apiService.getYahooFinanceTicker, symbol);
  }, [makeRequest]);

  const getOptions = useCallback(() => {
    return makeRequest(apiService.getOptions);
  }, [makeRequest]);

  const getOption = useCallback((id) => {
    return makeRequest(apiService.getOption, id);
  }, [makeRequest]);

  const createOption = useCallback((optionData) => {
    return makeRequest(apiService.createOption, optionData);
  }, [makeRequest]);

  const updateOption = useCallback((id, optionData) => {
    return makeRequest(apiService.updateOption, id, optionData);
  }, [makeRequest]);

  const deleteOption = useCallback((id) => {
    return makeRequest(apiService.deleteOption, id);
  }, [makeRequest]);

  const calculateOptionPrice = useCallback((optionData) => {
    return makeRequest(apiService.calculateOptionPrice, optionData);
  }, [makeRequest]);

  return {
    loading,
    error,
    data,
    clearState,
    getHealth,
    getYahooFinanceTest,
    getYahooFinanceTicker,
    getOptions,
    getOption,
    createOption,
    updateOption,
    deleteOption,
    calculateOptionPrice,
  };
};

// Hook específico para Portfolio Management
export const usePortfolioManagement = () => {
  const { loading, error, data, makeRequest, clearState } = useApi();

  const getHealth = useCallback(() => {
    return makeRequest(apiService.getPortfolioManagementHealth);
  }, [makeRequest]);

  const getPortfolios = useCallback(() => {
    return makeRequest(apiService.getPortfolios);
  }, [makeRequest]);

  const getPortfolio = useCallback((id) => {
    return makeRequest(apiService.getPortfolio, id);
  }, [makeRequest]);

  const createPortfolio = useCallback((portfolioData) => {
    return makeRequest(apiService.createPortfolio, portfolioData);
  }, [makeRequest]);

  const updatePortfolio = useCallback((id, portfolioData) => {
    return makeRequest(apiService.updatePortfolio, id, portfolioData);
  }, [makeRequest]);

  const deletePortfolio = useCallback((id) => {
    return makeRequest(apiService.deletePortfolio, id);
  }, [makeRequest]);

  return {
    loading,
    error,
    data,
    clearState,
    getHealth,
    getPortfolios,
    getPortfolio,
    createPortfolio,
    updatePortfolio,
    deletePortfolio,
  };
};

// Hook para verificar la salud de todos los servicios
export const useServicesHealth = () => {
  const { loading, error, data, makeRequest, clearState } = useApi();

  const checkAllServices = useCallback(() => {
    return makeRequest(apiService.checkAllServices);
  }, [makeRequest]);

  return {
    loading,
    error,
    data,
    clearState,
    checkAllServices,
  };
};
