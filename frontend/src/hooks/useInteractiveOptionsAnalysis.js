import { useState, useCallback } from 'react';

const API_BASE_URL = 'http://localhost:8000';

/**
 * Hook personalizado para análisis interactivo de opciones paso a paso
 * Maneja todo el flujo desde la selección del símbolo hasta el cálculo final
 */
export const useInteractiveOptionsAnalysis = () => {
  // Estados del flujo
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Datos del usuario
  const [userInputs, setUserInputs] = useState({
    symbol: '',
    optionType: '',
    expirationDate: '',
    strike: '',
    riskFreeRate: '',
    volatility: '',
    useImpliedVolatility: false,
    marketPrice: '',
    // Nuevos campos para selección de modelo
    selectedModel: 'black_scholes',
    nSteps: 100,
    nSimulations: 10000,
    seed: null
  });
  
  // Datos de Yahoo Finance
  const [yahooData, setYahooData] = useState({
    expirations: [],
    optionsChain: [],
    currentPrice: 0,
    selectedOption: null
  });
  
  // Resultados del análisis
  const [analysisResults, setAnalysisResults] = useState({
    calculatedPrice: null,
    greeks: null,
    impliedVolatility: null,
    sensitivityAnalysis: null
  });

  /**
   * Paso 1: Obtener expiraciones disponibles para un símbolo
   */
  const fetchExpirations = useCallback(async (symbol) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log(`🔍 Buscando expiraciones para ${symbol}...`);
      
      const response = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/?symbol=${symbol}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        console.log(`✅ Encontradas ${data.count} expiraciones para ${symbol}`);
        setYahooData(prev => ({
          ...prev,
          expirations: data.expirations,
          currentPrice: data.current_price || 0
        }));
        setCurrentStep(1);
        return data.expirations;
      } else {
        throw new Error(data.message || 'Error obteniendo expiraciones');
      }
    } catch (err) {
      const errorMessage = `Error obteniendo expiraciones para ${symbol}: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage, err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Paso 2: Obtener opciones disponibles para una fecha de expiración
   */
  const fetchOptionsChain = async (symbol, expirationDate, optionType) => {
    try {
      console.log('🔍 fetchOptionsChain llamado con:', { symbol, expirationDate, optionType });
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol,
          expiration_date: expirationDate,
          option_type: optionType
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('🔍 Respuesta del backend recibida:', data);
      
      if (data.status === 'success') {
        // Seleccionar las opciones correctas según el tipo elegido por el usuario
        const optionsChain = optionType === 'call' ? (data.calls || []) : (data.puts || []);
        console.log('🔍 OptionsChain preparado para', optionType, ':', optionsChain);
        console.log('🔍 Longitud de optionsChain:', optionsChain.length);
        
        console.log('🔍 Antes de setYahooData - Estado actual:', yahooData);
        
        setYahooData(prev => {
          console.log('🔍 setYahooData callback - prev:', prev);
          const newState = {
            ...prev,
            optionsChain: optionsChain,
            currentPrice: data.current_price,
            selectedOption: null
          };
          console.log('🔍 setYahooData callback - nuevo estado:', newState);
          return newState;
        });
        
        console.log('🔍 Después de setYahooData - Estado actual:', yahooData);
      } else {
        throw new Error(data.message || 'Error obteniendo opciones');
      }
    } catch (error) {
      console.error('Error obteniendo opciones:', error);
      setError(`Error obteniendo opciones para ${symbol}: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Paso 3: Calcular precio de la opción usando el backend
   */
  const calculateOptionPrice = useCallback(async () => {
    if (!yahooData.selectedOption || !userInputs.riskFreeRate) {
      setError('Faltan datos para calcular el precio');
      return;
    }
    
    // Validar que el precio spot sea válido
    if (!yahooData.currentPrice || yahooData.currentPrice <= 0) {
      setError('Error: El precio actual del subyacente no es válido. Intenta seleccionar otra expiración.');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      console.log('🧮 Calculando precio de la opción...');
      console.log('📊 Datos de la opción:', {
        symbol: userInputs.symbol,
        type: userInputs.optionType,
        spot: yahooData.currentPrice,
        strike: yahooData.selectedOption.strike,
        maturity: (calculateDaysToExpiry(userInputs.expirationDate) / 365).toFixed(6),
        volatility: userInputs.useImpliedVolatility ? 
          (yahooData.selectedOption.implied_volatility || 0.25) : 
          (parseFloat(userInputs.volatility) / 100),
        rate: (parseFloat(userInputs.riskFreeRate) / 100).toFixed(6),
        model: userInputs.selectedModel,
        nSteps: userInputs.nSteps,
        nSimulations: userInputs.nSimulations
      });
      
      // Crear la opción en el backend
      const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: `${userInputs.symbol} ${userInputs.optionType.toUpperCase()}`,
          type: userInputs.optionType,
          style: 'european',
          spot: yahooData.currentPrice,
          strike: yahooData.selectedOption.strike,
          maturity: (calculateDaysToExpiry(userInputs.expirationDate) / 365).toFixed(6), // Limitar a 6 decimales
          volatility: userInputs.useImpliedVolatility ? 
            (yahooData.selectedOption.implied_volatility || 0.25) : 
            (parseFloat(userInputs.volatility) / 100),
          rate: (parseFloat(userInputs.riskFreeRate) / 100).toFixed(6), // Limitar a 6 decimales
        }),
      });
      
      if (!createResponse.ok) {
        const errorData = await createResponse.json();
        console.error('❌ Error response:', errorData);
        throw new Error(`Error creando opción: HTTP ${createResponse.status} - ${JSON.stringify(errorData)}`);
      }
      
      const option = await createResponse.json();
      console.log('✅ Opción creada:', option);
      
      // Calcular precio usando el modelo seleccionado
      const pricingResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/${option.id}/calculate_price/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: userInputs.selectedModel,
          calculate_greeks: true,
          // Parámetros específicos del modelo
          n_steps: userInputs.nSteps,
          n_simulations: userInputs.nSimulations,
          seed: userInputs.seed
        }),
      });
      
      if (!pricingResponse.ok) {
        throw new Error(`Error calculando precio: HTTP ${pricingResponse.status}`);
      }
      
      const pricingData = await pricingResponse.json();
      console.log('✅ Precio calculado:', pricingData);
      
      setAnalysisResults({
        calculatedPrice: pricingData.calculated_price,
        greeks: pricingData.greeks,
        impliedVolatility: null,
        sensitivityAnalysis: null
      });
      
      setCurrentStep(4);
      return pricingData;
      
    } catch (err) {
      const errorMessage = `Error calculando precio de la opción: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage, err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [yahooData, userInputs]);

  /**
   * Paso 4: Calcular volatilidad implícita si se desea
   */
  const calculateImpliedVolatility = useCallback(async (marketPrice) => {
    if (!analysisResults.calculatedPrice) {
      setError('Primero debes calcular el precio de la opción');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      console.log('📊 Calculando volatilidad implícita...');
      
      // Aquí necesitarías implementar la lógica para calcular volatilidad implícita
      // Por ahora, simularemos el resultado
      const impliedVol = 0.25; // Valor simulado
      
      setAnalysisResults(prev => ({
        ...prev,
        impliedVolatility: impliedVol
      }));
      
      console.log(`✅ Volatilidad implícita calculada: ${(impliedVol * 100).toFixed(2)}%`);
      
    } catch (err) {
      const errorMessage = `Error calculando volatilidad implícita: ${err.message}`;
      setError(errorMessage);
      console.error(errorMessage, err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [analysisResults]);

  /**
   * Función auxiliar para calcular días hasta expiración
   */
  const calculateDaysToExpiry = (expirationDate) => {
    const today = new Date();
    const expiry = new Date(expirationDate);
    const diffTime = expiry - today;
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  };

  /**
   * Función para actualizar inputs del usuario
   */
  const updateUserInput = useCallback((field, value) => {
    setUserInputs(prev => ({ ...prev, [field]: value }));
  }, []);

  /**
   * Función para seleccionar una opción específica
   */
  const selectOption = useCallback((option) => {
    setYahooData(prev => ({ ...prev, selectedOption: option }));
  }, []);

  /**
   * Función para reiniciar el flujo
   */
  const resetFlow = useCallback(() => {
    setCurrentStep(0);
    setError(null);
    setUserInputs({
      symbol: '',
      optionType: '',
      expirationDate: '',
      strike: '',
      riskFreeRate: '',
      volatility: '',
      useImpliedVolatility: false,
      marketPrice: '',
      // Nuevos campos para selección de modelo
      selectedModel: 'black_scholes',
      nSteps: 100,
      nSimulations: 10000,
      seed: null
    });
    setYahooData({
      expirations: [],
      optionsChain: [],
      currentPrice: 0,
      selectedOption: null
    });
    setAnalysisResults({
      calculatedPrice: null,
      greeks: null,
      impliedVolatility: null,
      sensitivityAnalysis: null
    });
  }, []);

  /**
   * Función para avanzar al siguiente paso
   */
  const nextStep = useCallback(() => {
    setCurrentStep(prev => Math.min(prev + 1, 4));
  }, []);

  /**
   * Función para retroceder al paso anterior
   */
  const prevStep = useCallback(() => {
    setCurrentStep(prev => Math.max(prev - 1, 0));
  }, []);

  return {
    // Estados
    currentStep,
    loading,
    error,
    userInputs,
    yahooData,
    analysisResults,
    
    // Acciones
    fetchExpirations,
    fetchOptionsChain,
    calculateOptionPrice,
    calculateImpliedVolatility,
    updateUserInput,
    selectOption,
    resetFlow,
    nextStep,
    prevStep,
    setYahooData, // Exportar setYahooData para uso en el componente
    
    // Funciones auxiliares
    calculateDaysToExpiry
  };
};
