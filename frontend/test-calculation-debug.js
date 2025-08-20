// Test de Debugging para Cálculo de Precio
// Ejecutar en la consola del navegador para diagnosticar problemas

console.log('🔍 === TEST DE DEBUGGING PARA CÁLCULO DE PRECIO ===');

// Función para verificar el estado completo
function checkApplicationState() {
  console.log('📋 === ESTADO COMPLETO DE LA APLICACIÓN ===');
  
  // Verificar variables globales
  if (typeof currentStep !== 'undefined') {
    console.log('✅ Current Step:', currentStep);
  } else {
    console.log('❌ currentStep no está definido');
  }
  
  if (typeof userInputs !== 'undefined') {
    console.log('✅ User Inputs:', userInputs);
  } else {
    console.log('❌ userInputs no está definido');
  }
  
  if (typeof yahooData !== 'undefined') {
    console.log('✅ Yahoo Data:', yahooData);
  } else {
    console.log('❌ yahooData no está definido');
  }
  
  if (typeof analysisResults !== 'undefined') {
    console.log('✅ Analysis Results:', analysisResults);
  } else {
    console.log('❌ analysisResults no está definido');
  }
  
  if (typeof loading !== 'undefined') {
    console.log('✅ Loading:', loading);
  } else {
    console.log('❌ loading no está definido');
  }
  
  if (typeof error !== 'undefined') {
    console.log('✅ Error:', error);
  } else {
    console.log('❌ error no está definido');
  }
}

// Función para verificar la función de cálculo
function checkCalculationFunction() {
  console.log('🧮 === VERIFICANDO FUNCIÓN DE CÁLCULO ===');
  
  if (typeof calculateOptionPrice !== 'undefined') {
    console.log('✅ calculateOptionPrice está definida');
    console.log('📝 Tipo:', typeof calculateOptionPrice);
    console.log('📝 Es función:', typeof calculateOptionPrice === 'function');
  } else {
    console.log('❌ calculateOptionPrice no está definida');
  }
  
  if (typeof handleCalculatePrice !== 'undefined') {
    console.log('✅ handleCalculatePrice está definida');
    console.log('📝 Tipo:', typeof handleCalculatePrice);
    console.log('📝 Es función:', typeof handleCalculatePrice === 'function');
  } else {
    console.log('❌ handleCalculatePrice no está definida');
  }
}

// Función para verificar la función de validación
function checkValidationFunction() {
  console.log('🔍 === VERIFICANDO FUNCIÓN DE VALIDACIÓN ===');
  
  if (typeof validateAndFormatValue !== 'undefined') {
    console.log('✅ validateAndFormatValue está definida');
    
    // Probar la función con valores de ejemplo
    try {
      const testSpot = validateAndFormatValue(100, 'spot');
      console.log('✅ Test spot (100):', testSpot);
    } catch (error) {
      console.error('❌ Error en test spot:', error);
    }
    
    try {
      const testStrike = validateAndFormatValue(95, 'strike');
      console.log('✅ Test strike (95):', testStrike);
    } catch (error) {
      console.error('❌ Error en test strike:', error);
    }
    
    try {
      const testVolatility = validateAndFormatValue(0.25, 'volatility');
      console.log('✅ Test volatility (0.25):', testVolatility);
    } catch (error) {
      console.error('❌ Error en test volatility:', error);
    }
    
    try {
      const testRate = validateAndFormatValue(0.05, 'rate');
      console.log('✅ Test rate (0.05):', testRate);
    } catch (error) {
      console.error('❌ Error en test rate:', error);
    }
    
    try {
      const testMaturity = validateAndFormatValue(0.5, 'maturity');
      console.log('✅ Test maturity (0.5):', testMaturity);
    } catch (error) {
      console.error('❌ Error en test maturity:', error);
    }
    
  } else {
    console.log('❌ validateAndFormatValue no está definida');
  }
}

// Función para verificar la función de cálculo de madurez
function checkMaturityFunction() {
  console.log('📅 === VERIFICANDO FUNCIÓN DE MADUREZ ===');
  
  if (typeof calculateDaysToExpiry !== 'undefined') {
    console.log('✅ calculateDaysToExpiry está definida');
    
    // Probar con fechas de ejemplo
    try {
      const today = new Date();
      const futureDate = new Date(today.getTime() + (30 * 24 * 60 * 60 * 1000)); // 30 días en el futuro
      const days = calculateDaysToExpiry(futureDate.toISOString().split('T')[0]);
      console.log('✅ Test madurez (30 días):', days);
    } catch (error) {
      console.error('❌ Error en test madurez:', error);
    }
    
  } else {
    console.log('❌ calculateDaysToExpiry no está definida');
  }
}

// Función para verificar conectividad del backend
async function checkBackendConnectivity() {
  console.log('🌐 === VERIFICANDO CONECTIVIDAD DEL BACKEND ===');
  
  try {
    const response = await fetch('http://localhost:8000/api/option-pricing/health/');
    console.log('✅ Health check status:', response.status);
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ Health check data:', data);
    } else {
      console.log('⚠️ Health check no exitoso');
    }
  } catch (error) {
    console.error('❌ Error en health check:', error);
  }
  
  try {
    const response = await fetch('http://localhost:8000/api/option-pricing/options/');
    console.log('✅ Options endpoint status:', response.status);
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ Options endpoint data:', data);
    } else {
      console.log('⚠️ Options endpoint no exitoso');
    }
  } catch (error) {
    console.error('❌ Error en options endpoint:', error);
  }
}

// Función para simular el cálculo paso a paso
function simulateCalculation() {
  console.log('🧪 === SIMULANDO CÁLCULO PASO A PASO ===');
  
  // Verificar que tenemos los datos necesarios
  if (!yahooData || !userInputs) {
    console.log('❌ No hay datos disponibles para simular');
    return;
  }
  
  console.log('📊 Datos disponibles para simulación:');
  console.log('- Symbol:', userInputs.symbol);
  console.log('- Option Type:', userInputs.optionType);
  console.log('- Option Style:', userInputs.optionStyle);
  console.log('- Expiration Date:', userInputs.expirationDate);
  console.log('- Risk Free Rate:', userInputs.riskFreeRate);
  console.log('- Volatility:', userInputs.volatility);
  console.log('- Use Implied Volatility:', userInputs.useImpliedVolatility);
  console.log('- Selected Model:', userInputs.selectedModel);
  console.log('- Current Price:', yahooData.currentPrice);
  console.log('- Selected Option:', yahooData.selectedOption);
  
  // Simular validaciones
  console.log('🔍 Simulando validaciones...');
  
  if (!yahooData.selectedOption) {
    console.log('❌ No hay opción seleccionada');
    return;
  }
  
  if (!userInputs.riskFreeRate) {
    console.log('❌ No hay tasa libre de riesgo');
    return;
  }
  
  if (!yahooData.currentPrice || yahooData.currentPrice <= 0) {
    console.log('❌ Precio actual no válido');
    return;
  }
  
  if (!userInputs.expirationDate) {
    console.log('❌ No hay fecha de expiración');
    return;
  }
  
  console.log('✅ Todas las validaciones pasaron');
  
  // Simular cálculo de madurez
  try {
    const daysToExpiry = calculateDaysToExpiry(userInputs.expirationDate);
    console.log('📅 Días hasta expiración:', daysToExpiry);
    
    if (daysToExpiry <= 0) {
      console.log('❌ La fecha de expiración debe ser futura');
      return;
    }
    
    const maturityYears = daysToExpiry / 365;
    console.log('📅 Madurez en años:', maturityYears);
    
  } catch (error) {
    console.error('❌ Error calculando madurez:', error);
    return;
  }
  
  // Simular validación de valores
  try {
    console.log('🔍 Simulando validación de valores...');
    
    const spot = validateAndFormatValue(yahooData.currentPrice, 'spot');
    console.log('✅ Spot validado:', spot);
    
    const strike = validateAndFormatValue(yahooData.selectedOption.strike, 'strike');
    console.log('✅ Strike validado:', strike);
    
    const maturity = validateAndFormatValue(daysToExpiry / 365, 'maturity');
    console.log('✅ Maturity validado:', maturity);
    
    const volatilityValue = userInputs.useImpliedVolatility ? 
      (yahooData.selectedOption.implied_volatility || 0.25) : 
      (parseFloat(userInputs.volatility) / 100);
    const volatility = validateAndFormatValue(volatilityValue, 'volatility');
    console.log('✅ Volatility validado:', volatility);
    
    const rateValue = parseFloat(userInputs.riskFreeRate) / 100;
    const rate = validateAndFormatValue(rateValue, 'rate');
    console.log('✅ Rate validado:', rate);
    
    console.log('✅ Todos los valores validados exitosamente');
    
    // Simular payload para el backend
    const payload = {
      name: `${userInputs.symbol} ${userInputs.optionType.toUpperCase()}`,
      type: userInputs.optionType,
      style: userInputs.optionStyle,
      spot: spot,
      strike: strike,
      maturity: maturity,
      volatility: volatility,
      rate: rate,
    };
    
    console.log('📤 Payload simulado:', payload);
    
  } catch (error) {
    console.error('❌ Error en validación de valores:', error);
    return;
  }
  
  console.log('🎉 Simulación completada exitosamente');
}

// Función principal de testing
function runAllTests() {
  console.log('🚀 === EJECUTANDO TODOS LOS TESTS ===');
  
  checkApplicationState();
  checkCalculationFunction();
  checkValidationFunction();
  checkMaturityFunction();
  
  // Los tests asíncronos se ejecutan por separado
  setTimeout(() => {
    checkBackendConnectivity();
  }, 1000);
  
  setTimeout(() => {
    simulateCalculation();
  }, 2000);
}

// Exportar funciones para uso manual
window.debugCalculation = {
  checkApplicationState,
  checkCalculationFunction,
  checkValidationFunction,
  checkMaturityFunction,
  checkBackendConnectivity,
  simulateCalculation,
  runAllTests
};

console.log('✅ Funciones de debug disponibles en window.debugCalculation');
console.log('💡 Usa debugCalculation.runAllTests() para ejecutar todos los tests');
console.log('💡 O ejecuta funciones individuales como debugCalculation.checkApplicationState()');
