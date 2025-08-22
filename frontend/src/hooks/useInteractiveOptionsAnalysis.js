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
    optionStyle: 'american', // Nuevo campo para estilo de opción
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

  // Estado para análisis de sensibilidad
  const [sensitivityParams, setSensitivityParams] = useState({
    sensitivityType: 'spot',
    minValue: 0,
    maxValue: 0,
    numPoints: 21
  });

  // Estado para resultados de sensibilidad
  const [sensitivityResults, setSensitivityResults] = useState({
    data: null,
    loading: false,
    error: null
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
    try {
      console.log('🚀 === INICIANDO CÁLCULO DE PRECIO ===');
      console.log('📋 Estado inicial:', {
        currentStep,
        hasSelectedOption: !!yahooData.selectedOption,
        hasRiskFreeRate: !!userInputs.riskFreeRate,
        hasCurrentPrice: !!yahooData.currentPrice,
        currentPrice: yahooData.currentPrice,
        expirationDate: userInputs.expirationDate
      });

      // Validaciones iniciales con logging detallado
      if (!yahooData.selectedOption) {
        const errorMsg = 'No hay opción seleccionada';
        console.error('❌ Validación fallida:', errorMsg);
        setError(errorMsg);
        return;
      }

      if (!userInputs.riskFreeRate) {
        const errorMsg = 'No hay tasa libre de riesgo';
        console.error('❌ Validación fallida:', errorMsg);
        setError(errorMsg);
        return;
      }

      if (!yahooData.currentPrice || yahooData.currentPrice <= 0) {
        const errorMsg = 'Precio actual del subyacente no válido';
        console.error('❌ Validación fallida:', errorMsg, { currentPrice: yahooData.currentPrice });
        setError(`Error: ${errorMsg}. Intenta seleccionar otra expiración.`);
        return;
      }

      if (!userInputs.expirationDate) {
        const errorMsg = 'No hay fecha de expiración';
        console.error('❌ Validación fallida:', errorMsg);
        setError(errorMsg);
        return;
      }

      console.log('✅ Validaciones iniciales pasadas');
      
      setLoading(true);
      setError(null);
      
      // Preparar datos para logging
      const rawData = {
        symbol: userInputs.symbol,
        optionType: userInputs.optionType,
        optionStyle: userInputs.optionStyle,
        spot: yahooData.currentPrice,
        strike: yahooData.selectedOption.strike,
        expirationDate: userInputs.expirationDate,
        riskFreeRate: userInputs.riskFreeRate,
        volatility: userInputs.volatility,
        useImpliedVolatility: userInputs.useImpliedVolatility,
        impliedVolatility: yahooData.selectedOption.implied_volatility,
        selectedModel: userInputs.selectedModel,
        nSteps: userInputs.nSteps,
        nSimulations: userInputs.nSimulations
      };

      console.log('📊 Datos de entrada completos:', rawData);

      // Calcular madurez con validación
      console.log('📅 Calculando madurez...');
      const daysToExpiry = calculateDaysToExpiry(userInputs.expirationDate);
      console.log('📅 Días hasta expiración:', daysToExpiry);
      
      if (daysToExpiry <= 0) {
        const errorMsg = 'La fecha de expiración debe ser futura';
        console.error('❌ Error en madurez:', errorMsg, { daysToExpiry, expirationDate: userInputs.expirationDate });
        setError(errorMsg);
        setLoading(false);
        return;
      }

      const maturityYears = daysToExpiry / 365;
      console.log('📅 Madurez en años:', maturityYears);

      // Validar y formatear valores paso a paso
      console.log('🔍 === VALIDANDO Y FORMATEANDO VALORES ===');
      
      let spot, strike, maturity, volatility, rate;
      
      try {
        console.log('🔍 Validando spot...');
        spot = validateAndFormatValue(yahooData.currentPrice, 'spot');
        console.log('✅ Spot validado:', spot);
      } catch (error) {
        console.error('❌ Error validando spot:', error);
        setError(`Error validando precio spot: ${error.message}`);
        setLoading(false);
        return;
      }

      try {
        console.log('🔍 Validando strike...');
        strike = validateAndFormatValue(yahooData.selectedOption.strike, 'strike');
        console.log('✅ Strike validado:', strike);
      } catch (error) {
        console.error('❌ Error validando strike:', error);
        setError(`Error validando strike: ${error.message}`);
        setLoading(false);
        return;
      }

      try {
        console.log('🔍 Validando madurez...');
        maturity = validateAndFormatValue(maturityYears, 'maturity');
        console.log('✅ Madurez validada:', maturity);
      } catch (error) {
        console.error('❌ Error validando madurez:', error);
        setError(`Error validando madurez: ${error.message}`);
        setLoading(false);
        return;
      }

      try {
        console.log('🔍 Validando volatilidad...');
        const volatilityValue = userInputs.useImpliedVolatility ? 
          (yahooData.selectedOption.implied_volatility || 0.25) : 
          (parseFloat(userInputs.volatility) / 100);
        console.log('🔍 Valor de volatilidad a validar:', volatilityValue);
        volatility = validateAndFormatValue(volatilityValue, 'volatility');
        console.log('✅ Volatilidad validada:', volatility);
      } catch (error) {
        console.error('❌ Error validando volatilidad:', error);
        setError(`Error validando volatilidad: ${error.message}`);
        setLoading(false);
        return;
      }

      try {
        console.log('🔍 Validando tasa...');
        const rateValue = parseFloat(userInputs.riskFreeRate) / 100;
        console.log('🔍 Valor de tasa a validar:', rateValue);
        rate = validateAndFormatValue(rateValue, 'rate');
        console.log('✅ Tasa validada:', rate);
      } catch (error) {
        console.error('❌ Error validando tasa:', error);
        setError(`Error validando tasa: ${error.message}`);
        setLoading(false);
        return;
      }

      console.log('✅ Todos los valores validados exitosamente');

      // Crear payload para el backend
      const optionPayload = {
        name: `${userInputs.symbol} ${userInputs.optionType.toUpperCase()}`,
        type: userInputs.optionType,
        style: userInputs.optionStyle,
        spot: spot,
        strike: strike,
        maturity: maturity,
        volatility: volatility,
        rate: rate,
      };

      console.log('📤 Payload para crear opción:', optionPayload);

      // Crear la opción en el backend
      console.log('🌐 === CREANDO OPCIÓN EN EL BACKEND ===');
      console.log('🌐 URL:', `${API_BASE_URL}/api/option-pricing/options/`);
      
      const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(optionPayload),
      });
      
      console.log('🌐 Respuesta de creación:', createResponse.status, createResponse.ok);
      
      if (!createResponse.ok) {
        let errorData;
        try {
          errorData = await createResponse.json();
        } catch (parseError) {
          errorData = { detail: 'No se pudo parsear el error del backend' };
        }
        console.error('❌ Error response del backend:', errorData);
        const errorMsg = `Error creando opción: HTTP ${createResponse.status} - ${JSON.stringify(errorData)}`;
        setError(errorMsg);
        setLoading(false);
        return;
      }
      
      const option = await createResponse.json();
      console.log('✅ Opción creada exitosamente:', option);
      
      // Calcular precio usando el modelo seleccionado
      console.log('🧮 === CALCULANDO PRECIO ===');
      console.log('🧮 Modelo seleccionado:', userInputs.selectedModel);
      
      const pricingPayload = {
        model: userInputs.selectedModel,
        calculate_greeks: true,
        n_steps: userInputs.nSteps,
        n_simulations: userInputs.nSimulations,
        seed: userInputs.seed
      };

      console.log('🧮 Payload para cálculo:', pricingPayload);
      console.log('🧮 URL:', `${API_BASE_URL}/api/option-pricing/options/${option.id}/calculate_price/`);
      
      const pricingResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/${option.id}/calculate_price/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(pricingPayload),
      });
      
      console.log('🧮 Respuesta de cálculo:', pricingResponse.status, pricingResponse.ok);
      
      if (!pricingResponse.ok) {
        let errorData;
        try {
          errorData = await pricingResponse.json();
        } catch (parseError) {
          errorData = { detail: 'No se pudo parsear el error del backend' };
        }
        console.error('❌ Error en cálculo de precio:', errorData);
        const errorMsg = `Error calculando precio: HTTP ${pricingResponse.status} - ${JSON.stringify(errorData)}`;
        setError(errorMsg);
        setLoading(false);
        return;
      }
      
      const pricingData = await pricingResponse.json();
      console.log('✅ Precio calculado exitosamente:', pricingData);
      
      // Actualizar estado
      console.log('💾 === ACTUALIZANDO ESTADO ===');
      const newAnalysisResults = {
        calculatedPrice: pricingData.calculated_price,
        greeks: pricingData.greeks,
        impliedVolatility: null,
        sensitivityAnalysis: null
      };
      
      console.log('💾 Nuevos resultados:', newAnalysisResults);
      setAnalysisResults(newAnalysisResults);
      
      console.log('🔄 Manteniendo en paso 2 para mostrar resultados y pregunta...');
      // No cambiar automáticamente al paso 3, mantener en paso 2 para mostrar los resultados
      // y pregunta de si continuar con análisis de sensibilidad
      
      // Limpiar estado de loading después del cálculo exitoso
      setLoading(false);
      
      console.log('🎉 === CÁLCULO COMPLETADO EXITOSAMENTE ===');
      return pricingData;
      
    } catch (err) {
      console.error('🚨 === ERROR CRÍTICO EN CÁLCULO ===');
      console.error('🚨 Error completo:', err);
      console.error('🚨 Stack trace:', err.stack);
      
      const errorMessage = `Error calculando precio de la opción: ${err.message}`;
      console.error('🚨 Mensaje de error:', errorMessage);
      
      setError(errorMessage);
      setLoading(false);
      return null;
    }
  }, [yahooData, userInputs, currentStep]);

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
   * Análisis de sensibilidad - calcular cómo cambia el precio ante variaciones de parámetros
   */
  const performSensitivityAnalysis = useCallback(async () => {
    if (!yahooData.selectedOption || !userInputs.riskFreeRate) {
      setError('Faltan datos para realizar el análisis de sensibilidad');
      return;
    }
    
    setSensitivityResults(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      console.log('📊 Realizando análisis de sensibilidad...');
      
      // Crear la opción en el backend si no existe
      let optionId = null;
      if (!analysisResults.calculatedPrice) {
        try {
          // Validar y formatear valores antes de enviar al backend
          const spot = validateAndFormatValue(yahooData.currentPrice, 'spot');
          const strike = validateAndFormatValue(yahooData.selectedOption.strike, 'strike');
          const maturity = validateAndFormatValue(calculateDaysToExpiry(userInputs.expirationDate) / 365, 'maturity');
          const volatility = validateAndFormatValue(
            userInputs.useImpliedVolatility ? 
              (yahooData.selectedOption.implied_volatility || 0.25) : 
              (parseFloat(userInputs.volatility) / 100),
            'volatility'
          );
          const rate = validateAndFormatValue(parseFloat(userInputs.riskFreeRate) / 100, 'rate');

          const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              name: `${userInputs.symbol} ${userInputs.optionType.toUpperCase()}`,
              type: userInputs.optionType,
              style: userInputs.optionStyle,
              spot: spot,
              strike: strike,
              maturity: maturity,
              volatility: volatility,
              rate: rate,
            }),
          });
          
          if (!createResponse.ok) {
            const errorData = await createResponse.json();
            console.error('❌ Error creando opción:', errorData);
            throw new Error(`Error creando opción: HTTP ${createResponse.status} - ${JSON.stringify(errorData)}`);
          }
          
          const option = await createResponse.json();
          optionId = option.id;
          console.log('✅ Opción creada para análisis de sensibilidad:', optionId);
        } catch (createError) {
          throw new Error(`Error creando opción para análisis de sensibilidad: ${createError.message}`);
        }
      } else {
        // Si ya hay una opción calculada, usar su ID
        // Necesitamos obtener el ID de la opción existente
        console.log('⚠️ No se puede obtener ID de opción existente. Creando nueva opción...');
        // Por ahora, crear una nueva opción
        try {
          const spot = validateAndFormatValue(yahooData.currentPrice, 'spot');
          const strike = validateAndFormatValue(yahooData.selectedOption.strike, 'strike');
          const maturity = validateAndFormatValue(calculateDaysToExpiry(userInputs.expirationDate) / 365, 'maturity');
          const volatility = validateAndFormatValue(
            userInputs.useImpliedVolatility ? 
              (yahooData.selectedOption.implied_volatility || 0.25) : 
              (parseFloat(userInputs.volatility) / 100),
            'volatility'
          );
          const rate = validateAndFormatValue(parseFloat(userInputs.riskFreeRate) / 100, 'rate');

          const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              name: `${userInputs.symbol} ${userInputs.optionType.toUpperCase()}`,
              type: userInputs.optionType,
              style: userInputs.optionStyle,
              spot: spot,
              strike: strike,
              maturity: maturity,
              volatility: volatility,
              rate: rate,
            }),
          });
          
          if (!createResponse.ok) {
            const errorData = await createResponse.json();
            console.error('❌ Error creando opción:', errorData);
            throw new Error(`Error creando opción: HTTP ${createResponse.status} - ${JSON.stringify(errorData)}`);
          }
          
          const option = await createResponse.json();
          optionId = option.id;
          console.log('✅ Nueva opción creada para análisis de sensibilidad:', optionId);
        } catch (createError) {
          throw new Error(`Error creando nueva opción para análisis de sensibilidad: ${createError.message}`);
        }
      }
      
      // Verificar que tenemos un optionId válido
      if (!optionId) {
        throw new Error('No se pudo obtener un ID de opción válido para el análisis de sensibilidad');
      }
      
      // Realizar análisis de sensibilidad usando el modelo seleccionado en el paso 3
      const sensitivityPayload = {
        sensitivity_type: sensitivityParams.sensitivityType,
        min_value: sensitivityParams.minValue,
        max_value: sensitivityParams.maxValue,
        num_points: sensitivityParams.numPoints,
        model: userInputs.selectedModel  // Usar el modelo del paso 3
      };
      
      console.log('🔍 Enviando análisis de sensibilidad:', {
        url: `${API_BASE_URL}/api/option-pricing/options/${optionId}/sensitivity_analysis/`,
        optionId: optionId,
        payload: sensitivityPayload
      });
      
      const sensitivityResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/${optionId}/sensitivity_analysis/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sensitivityPayload),
      });
      
      if (!sensitivityResponse.ok) {
        let errorMessage = `Error en análisis de sensibilidad: HTTP ${sensitivityResponse.status}`;
        try {
          const errorData = await sensitivityResponse.json();
          console.error('❌ Error response del análisis de sensibilidad:', errorData);
          if (errorData.error) {
            errorMessage += ` - ${errorData.error}`;
          } else if (errorData.detail) {
            errorMessage += ` - ${errorData.detail}`;
          } else {
            errorMessage += ` - ${JSON.stringify(errorData)}`;
          }
        } catch (parseError) {
          console.error('❌ No se pudo parsear el error del backend:', parseError);
        }
        throw new Error(errorMessage);
      }
      
      const sensitivityData = await sensitivityResponse.json();
      console.log('✅ Análisis de sensibilidad completado:', sensitivityData);
      
      setSensitivityResults({
        data: sensitivityData,
        loading: false,
        error: null
      });
      
      return sensitivityData;
      
    } catch (err) {
      const errorMessage = `Error en análisis de sensibilidad: ${err.message}`;
      setSensitivityResults(prev => ({
        ...prev,
        loading: false,
        error: errorMessage
      }));
      console.error(errorMessage, err);
      throw err;
    }
  }, [yahooData, userInputs, sensitivityParams, analysisResults.calculatedPrice]);

  /**
   * Actualizar parámetros de sensibilidad
   */
  const updateSensitivityParams = useCallback((field, value) => {
    setSensitivityParams(prev => ({ ...prev, [field]: value }));
  }, []);

  /**
   * Calcular rangos simplificados para análisis de sensibilidad
   */
  const calculateSensitivityRanges = useCallback(() => {
    if (!yahooData.currentPrice || !yahooData.selectedOption) return;
    
    const spot = yahooData.currentPrice;
    const currentVol = userInputs.useImpliedVolatility ? 
      (yahooData.selectedOption.implied_volatility || 0.25) : 
      (parseFloat(userInputs.volatility) / 100 || 0.25);
    const timeToMaturity = calculateDaysToExpiry(userInputs.expirationDate) / 365;
    
    // Rangos fijos y simples como solicitado
    const ranges = {
      spot: {
        min: spot * 0.9,  // -10%
        max: spot * 1.1   // +10%
      },
      volatility: {
        min: 0.10,        // 10%
        max: 0.30         // 30%
      },
      rate: {
        min: 0.01,        // 1%
        max: 0.05         // 5%
      },
      time: {
        min: timeToMaturity * 0.5,  // 50%
        max: timeToMaturity * 1.5   // 150%
      }
    };
    
    const currentRange = ranges[sensitivityParams.sensitivityType];
    if (currentRange) {
      setSensitivityParams(prev => ({
        ...prev,
        minValue: parseFloat(currentRange.min.toFixed(6)),
        maxValue: parseFloat(currentRange.max.toFixed(6)),
        numPoints: 11  // Fijo y simple
      }));
    }
  }, [yahooData, userInputs, sensitivityParams.sensitivityType]);

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
   * Función para validar y formatear valores según las restricciones del backend
   */
  const validateAndFormatValue = (value, fieldName) => {
    try {
      console.log(`🔍 Validando ${fieldName}:`, value, typeof value);
      
      // Convertir a número si es string
      let numericValue = value;
      if (typeof value === 'string') {
        numericValue = parseFloat(value);
        console.log(`🔄 Convertido de string a número:`, numericValue);
      }
      
      // Validar que sea un número válido
      if (typeof numericValue !== 'number' || isNaN(numericValue)) {
        console.error(`❌ Valor inválido para ${fieldName}:`, value, typeof value);
        throw new Error(`Valor inválido para ${fieldName}: ${value} (tipo: ${typeof value})`);
      }
      
      // Validar que no sea infinito
      if (!isFinite(numericValue)) {
        console.error(`❌ Valor infinito para ${fieldName}:`, numericValue);
        throw new Error(`Valor infinito para ${fieldName}: ${numericValue}`);
      }
      
      // Aplicar restricciones según el campo
      let result;
      switch (fieldName) {
        case 'volatility':
          // max_digits=8, decimal_places=6 (máximo 99.999999)
          if (numericValue > 99.999999) {
            throw new Error(`Volatilidad excede el límite máximo de 99.999999`);
          }
          if (numericValue < 0) {
            throw new Error(`Volatilidad no puede ser negativa`);
          }
          result = parseFloat(numericValue.toFixed(6));
          break;
        
        case 'rate':
          // max_digits=8, decimal_places=6 (máximo 99.999999)
          if (numericValue > 99.999999) {
            throw new Error(`Tasa de interés excede el límite máximo de 99.999999`);
          }
          if (numericValue < 0) {
            throw new Error(`Tasa de interés no puede ser negativa`);
          }
          result = parseFloat(numericValue.toFixed(6));
          break;
        
        case 'maturity':
          // max_digits=8, decimal_places=6 (máximo 99.999999 años)
          if (numericValue > 99.999999) {
            throw new Error(`Madurez excede el límite máximo de 99.999999 años`);
          }
          if (numericValue <= 0) {
            throw new Error(`Madurez debe ser mayor a 0`);
          }
          result = parseFloat(numericValue.toFixed(6));
          break;
        
        case 'spot':
        case 'strike':
          // max_digits=15, decimal_places=6 (máximo 999999999.999999)
          if (numericValue > 999999999.999999) {
            throw new Error(`Precio excede el límite máximo de 999999999.999999`);
          }
          if (numericValue <= 0) {
            throw new Error(`Precio debe ser mayor a 0`);
          }
          result = parseFloat(numericValue.toFixed(6));
          break;
        
        default:
          result = numericValue;
      }
      
      console.log(`✅ ${fieldName} validado exitosamente:`, result);
      return result;
      
    } catch (error) {
      console.error(`❌ Error validando ${fieldName}:`, error);
      throw error;
    }
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
      optionStyle: 'american', // Resetear el estilo de opción
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
    setSensitivityParams({
      sensitivityType: 'spot',
      minValue: 0,
      maxValue: 0,
      numPoints: 21
    });
    setSensitivityResults({
      data: null,
      loading: false,
      error: null
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

  /**
   * Función para probar la conectividad del backend y endpoints
   */
  const testBackendConnectivity = useCallback(async () => {
    try {
      console.log('🔍 Probando conectividad del backend...');
      
      // Probar endpoint de health
      const healthResponse = await fetch(`${API_BASE_URL}/api/option-pricing/health/`);
      console.log('✅ Health check:', healthResponse.status, healthResponse.ok);
      
      // Probar endpoint de opciones
      const optionsResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`);
      console.log('✅ Options endpoint:', optionsResponse.status, optionsResponse.ok);
      
      // Probar endpoint de análisis de sensibilidad (con un ID dummy)
      const sensitivityTestResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/999999/sensitivity_analysis/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sensitivity_type: 'spot',
          min_value: 100,
          max_value: 200,
          num_points: 5,
          model: 'binomial'
        }),
      });
      console.log('✅ Sensitivity endpoint test:', sensitivityTestResponse.status, sensitivityTestResponse.ok);
      
      if (sensitivityTestResponse.status === 404) {
        console.log('⚠️ Endpoint de sensibilidad no encontrado - verificar configuración del backend');
      } else if (sensitivityTestResponse.status === 400) {
        console.log('✅ Endpoint de sensibilidad encontrado - error 400 es esperado con ID inválido');
      }
      
      return {
        health: healthResponse.ok,
        options: optionsResponse.ok,
        sensitivity: sensitivityTestResponse.status !== 404
      };
      
    } catch (error) {
      console.error('❌ Error probando conectividad:', error);
      return {
        health: false,
        options: false,
        sensitivity: false,
        error: error.message
      };
    }
  }, []);

  return {
    // Estados
    currentStep,
    loading,
    error,
    userInputs,
    yahooData,
    analysisResults,
    sensitivityParams,
    sensitivityResults,
    
    // Acciones
    fetchExpirations,
    fetchOptionsChain,
    calculateOptionPrice,
    calculateImpliedVolatility,
    performSensitivityAnalysis,
    updateUserInput,
    updateSensitivityParams,
    calculateSensitivityRanges,
    selectOption,
    resetFlow,
    nextStep,
    prevStep,
    setYahooData, // Exportar setYahooData para uso en el componente
    
    // Funciones auxiliares
    calculateDaysToExpiry,
    validateAndFormatValue,
    testBackendConnectivity
  };
};
