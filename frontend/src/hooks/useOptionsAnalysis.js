import { useState, useCallback } from 'react';

const API_BASE_URL = 'http://localhost:8000';

export const useOptionsAnalysis = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [optionData, setOptionData] = useState(null);
  const [pricingResults, setPricingResults] = useState(null);
  const [greeks, setGreeks] = useState(null);
  const [currentOption, setCurrentOption] = useState(null);

  // Descargar datos de Yahoo Finance
  const fetchYahooFinanceData = useCallback(async (symbol) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/ticker/${symbol}/`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setOptionData(data);
      return data;
    } catch (err) {
      const errorMessage = `Error descargando datos de ${symbol}: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage, err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Calcular precio de la opción
  const calculateOptionPrice = useCallback(async (optionParams) => {
    setLoading(true);
    setError(null);
    
    try {
      // Paso 1: Crear la opción en el backend
      console.log('Creando opción con parámetros:', optionParams);
      
      const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: `${optionParams.symbol} ${optionParams.type.toUpperCase()}`,
          type: optionParams.type,
          style: 'european',
          spot: parseFloat(optionParams.spot),
          strike: parseFloat(optionParams.strike),
          maturity: parseFloat(optionParams.maturity),
          volatility: parseFloat(optionParams.volatility) / 100, // Convertir de % a decimal
          rate: parseFloat(optionParams.rate) / 100, // Convertir de % a decimal
        }),
      });
      
      if (!createResponse.ok) {
        const errorText = await createResponse.text();
        console.error('Error creando opción:', createResponse.status, errorText);
        throw new Error(`Error creando opción: HTTP ${createResponse.status}`);
      }
      
      const option = await createResponse.json();
      console.log('Opción creada:', option);
      setCurrentOption(option);
      
      // Paso 2: Calcular el precio usando el modelo seleccionado
      console.log('Calculando precio con modelo:', optionParams.model);
      
      const pricingResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/${option.id}/calculate_price/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: optionParams.model,
          calculate_greeks: true,
          n_steps: parseInt(optionParams.nSteps) || 100,
          n_simulations: parseInt(optionParams.nSimulations) || 10000,
        }),
      });
      
      if (!pricingResponse.ok) {
        const errorText = await pricingResponse.text();
        console.error('Error calculando precio:', pricingResponse.status, errorText);
        throw new Error(`Error calculando precio: HTTP ${pricingResponse.status}`);
      }
      
      const pricingData = await pricingResponse.json();
      console.log('Resultado del pricing:', pricingData);
      
      // Crear un objeto con la estructura esperada por el frontend
      const formattedPricingData = {
        calculated_price: pricingData.calculated_price,
        model: pricingData.model,
        option: option,
        greeks: pricingData.greeks,
        calculation_time: pricingData.calculation_time
      };
      
      setPricingResults(formattedPricingData);
      
      // Extraer griegas si están disponibles
      if (pricingData.greeks) {
        setGreeks(pricingData.greeks);
      }
      
      return { option, pricing: formattedPricingData };
    } catch (err) {
      const errorMessage = `Error calculando precio de la opción: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage, err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Calcular volatilidad implícita
  const calculateImpliedVolatility = useCallback(async (marketPrice) => {
    if (!currentOption?.id) {
      setError('No hay opción creada para calcular volatilidad implícita');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/option-pricing/options/${currentOption.id}/implied_volatility/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          market_price: parseFloat(marketPrice),
        }),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error calculando volatilidad implícita:', response.status, errorText);
        throw new Error(`Error calculando volatilidad implícita: HTTP ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (err) {
      const errorMessage = `Error calculando volatilidad implícita: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage, err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [currentOption]);

  // Análisis de sensibilidad
  const performSensitivityAnalysis = useCallback(async (analysisParams) => {
    if (!currentOption?.id) {
      setError('No hay opción creada para análisis de sensibilidad');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/option-pricing/options/${currentOption.id}/sensitivity_analysis/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(analysisParams),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error en análisis de sensibilidad:', response.status, errorText);
        throw new Error(`Error en análisis de sensibilidad: HTTP ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (err) {
      const errorMessage = `Error en análisis de sensibilidad: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage, err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [currentOption]);

  // Limpiar estado
  const clearState = useCallback(() => {
    setLoading(false);
    setError(null);
    setOptionData(null);
    setPricingResults(null);
    setGreeks(null);
    setCurrentOption(null);
  }, []);

  return {
    // Estado
    loading,
    error,
    optionData,
    pricingResults,
    greeks,
    currentOption,
    
    // Acciones
    fetchYahooFinanceData,
    calculateOptionPrice,
    calculateImpliedVolatility,
    performSensitivityAnalysis,
    clearState,
  };
};
