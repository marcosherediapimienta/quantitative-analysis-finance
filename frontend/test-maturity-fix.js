const API_BASE_URL = 'http://localhost:8000';

async function testMaturityFix() {
  console.log('🧪 Probando corrección del maturity...\n');
  
  try {
    // Paso 1: Obtener expiraciones para AAPL
    console.log('📅 Paso 1: Obteniendo expiraciones para AAPL...');
    const expirationsResponse = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/?symbol=AAPL`);
    
    if (!expirationsResponse.ok) {
      throw new Error(`Error obteniendo expiraciones: HTTP ${expirationsResponse.status}`);
    }
    
    const expirationsData = await expirationsResponse.json();
    console.log(`✅ Expiraciones obtenidas: ${expirationsData.count} fechas disponibles`);
    
    if (expirationsData.expirations.length === 0) {
      throw new Error('No se encontraron expiraciones');
    }
    
    // Seleccionar una expiración más lejana (al menos 30 días)
    let selectedExpiration = null;
    for (const exp of expirationsData.expirations) {
      if (exp.days_to_expiry >= 30) {
        selectedExpiration = exp;
        break;
      }
    }
    
    if (!selectedExpiration) {
      selectedExpiration = expirationsData.expirations[0];
    }
    
    console.log(`📅 Expiración seleccionada: ${selectedExpiration.formatted} (${selectedExpiration.date}) - ${selectedExpiration.days_to_expiry} días`);
    
    // Calcular maturity manualmente para verificar
    const daysToExpiry = selectedExpiration.days_to_expiry;
    const maturityYears = daysToExpiry / 365;
    const maturityFormatted = maturityYears.toFixed(6);
    
    console.log(`📊 Cálculo del maturity:`);
    console.log(`   - Días hasta expiración: ${daysToExpiry}`);
    console.log(`   - Años: ${maturityYears}`);
    console.log(`   - Formateado (6 decimales): ${maturityFormatted}`);
    console.log(`   - Total dígitos: ${maturityFormatted.replace('.', '').length}`);
    
    // Paso 2: Obtener opciones ATM para la expiración seleccionada
    console.log('\n🎯 Paso 2: Obteniendo opciones ATM...');
    const optionsResponse = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol: 'AAPL',
        expiration_date: selectedExpiration.date,
        option_type: 'atm'
      }),
    });
    
    if (!optionsResponse.ok) {
      throw new Error(`Error obteniendo opciones: HTTP ${optionsResponse.status}`);
    }
    
    const optionsData = await optionsResponse.json();
    console.log(`✅ Opciones obtenidas: ${optionsData.atm_options.length} opciones ATM`);
    
    if (optionsData.atm_options.length === 0) {
      throw new Error('No se encontraron opciones ATM');
    }
    
    // Mostrar información de la primera opción
    const selectedOption = optionsData.atm_options[0];
    console.log(`🎯 Opción seleccionada:`);
    console.log(`   - Strike: $${selectedOption.strike}`);
    console.log(`   - Precio: $${selectedOption.last_price}`);
    console.log(`   - Volatilidad implícita: ${(selectedOption.implied_volatility * 100).toFixed(2)}%`);
    console.log(`   - Volumen: ${selectedOption.volume}`);
    
    // Verificar el precio actual
    console.log(`\n💰 Precio actual del subyacente: $${optionsData.current_price}`);
    
    if (!optionsData.current_price || optionsData.current_price <= 0) {
      console.error('❌ ERROR: El precio actual no es válido');
      return;
    }
    
    console.log('✅ El precio actual es válido, continuando...');
    
    // Paso 3: Crear opción en el backend
    console.log('\n🧮 Paso 3: Creando opción en el backend...');
    
    // Preparar los datos con el formato correcto
    const optionData = {
      name: 'AAPL CALL Test Maturity Fix',
      type: 'call',
      style: 'european',
      spot: optionsData.current_price.toFixed(6),
      strike: selectedOption.strike.toFixed(6),
      maturity: maturityFormatted, // Usar el valor ya formateado
      volatility: (selectedOption.implied_volatility || 0.25).toFixed(6),
      rate: '0.050000',
    };
    
    console.log('📤 Datos enviados:', optionData);
    
    const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(optionData),
    });
    
    if (!createResponse.ok) {
      const errorData = await createResponse.json();
      console.error('❌ Error response:', errorData);
      throw new Error(`Error creando opción: HTTP ${createResponse.status} - ${JSON.stringify(errorData)}`);
    }
    
    const createdOption = await createResponse.json();
    console.log(`✅ Opción creada con ID: ${createdOption.id}`);
    
    // Paso 4: Calcular precio de la opción
    console.log('\n💰 Paso 4: Calculando precio de la opción...');
    const pricingResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/${createdOption.id}/calculate_price/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'black_scholes',
        calculate_greeks: true,
      }),
    });
    
    if (!pricingResponse.ok) {
      throw new Error(`Error calculando precio: HTTP ${pricingResponse.status}`);
    }
    
    const pricingData = await pricingResponse.json();
    console.log(`✅ Precio calculado: $${pricingData.calculated_price.toFixed(4)}`);
    
    if (pricingData.greeks) {
      console.log(`📊 Griegas calculadas:`);
      console.log(`   - Delta: ${parseFloat(pricingData.greeks.delta).toFixed(4)}`);
      console.log(`   - Gamma: ${parseFloat(pricingData.greeks.gamma).toFixed(4)}`);
      console.log(`   - Vega: ${parseFloat(pricingData.greeks.vega).toFixed(4)}`);
      console.log(`   - Theta: ${parseFloat(pricingData.greeks.theta).toFixed(4)}`);
      console.log(`   - Rho: ${parseFloat(pricingData.greeks.rho).toFixed(4)}`);
    }
    
    // Resumen final
    console.log('\n🎉 ¡Corrección del maturity completada exitosamente!');
    console.log(`📈 Símbolo: AAPL`);
    console.log(`📅 Expiración: ${selectedExpiration.formatted}`);
    console.log(`🎯 Strike: $${selectedOption.strike}`);
    console.log(`💰 Precio de mercado: $${selectedOption.last_price}`);
    console.log(`🧮 Precio calculado: $${pricingData.calculated_price.toFixed(4)}`);
    console.log(`📊 Diferencia: ${(((pricingData.calculated_price - selectedOption.last_price) / selectedOption.last_price) * 100).toFixed(2)}%`);
    console.log(`⏰ Maturity: ${maturityFormatted} años (${daysToExpiry} días)`);
    
  } catch (error) {
    console.error('❌ Error en la prueba:', error.message);
    if (error.response) {
      console.error('Respuesta del servidor:', error.response);
    }
  }
}

// Ejecutar la prueba
testMaturityFix();
