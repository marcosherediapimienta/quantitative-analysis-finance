const API_BASE_URL = 'http://localhost:8000';

async function testModelSelection() {
  console.log('🧪 Probando selección de modelos...\n');
  
  const models = [
    { name: 'Black-Scholes', value: 'black_scholes', params: {} },
    { name: 'Binomial', value: 'binomial', params: { n_steps: 200 } },
    { name: 'Monte Carlo', value: 'monte_carlo', params: { n_simulations: 50000, n_steps: 50 } }
  ];
  
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
    
    // Probar cada modelo
    for (const model of models) {
      console.log(`\n🧮 Probando modelo: ${model.name}...`);
      
      // Paso 3: Crear opción en el backend
      console.log('📝 Creando opción...');
      
      const optionData = {
        name: `AAPL CALL Test ${model.name}`,
        type: 'call',
        style: 'european',
        spot: optionsData.current_price.toFixed(6),
        strike: selectedOption.strike.toFixed(6),
        maturity: (selectedExpiration.days_to_expiry / 365).toFixed(6),
        volatility: (selectedOption.implied_volatility || 0.25).toFixed(6),
        rate: '0.050000',
      };
      
      const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(optionData),
      });
      
      if (!createResponse.ok) {
        const errorData = await createResponse.json();
        console.error(`❌ Error creando opción para ${model.name}:`, errorData);
        continue;
      }
      
      const createdOption = await createResponse.json();
      console.log(`✅ Opción creada con ID: ${createdOption.id}`);
      
      // Paso 4: Calcular precio usando el modelo específico
      console.log(`💰 Calculando precio con ${model.name}...`);
      
      const pricingData = {
        model: model.value,
        calculate_greeks: true,
        ...model.params
      };
      
      console.log('📤 Parámetros del modelo:', pricingData);
      
      const pricingResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/${createdOption.id}/calculate_price/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(pricingData),
      });
      
      if (!pricingResponse.ok) {
        console.error(`❌ Error calculando precio con ${model.name}: HTTP ${pricingResponse.status}`);
        continue;
      }
      
      const pricingResult = await pricingResponse.json();
      console.log(`✅ ${model.name} - Precio calculado: $${pricingResult.calculated_price.toFixed(4)}`);
      console.log(`   ⏱️  Tiempo de cálculo: ${pricingResult.calculation_time.toFixed(3)}s`);
      
      if (pricingResult.greeks) {
        console.log(`   📊 Delta: ${parseFloat(pricingResult.greeks.delta).toFixed(4)}`);
      }
    }
    
    // Resumen final
    console.log('\n🎉 ¡Prueba de selección de modelos completada exitosamente!');
    console.log(`📈 Símbolo: AAPL`);
    console.log(`📅 Expiración: ${selectedExpiration.formatted}`);
    console.log(`🎯 Strike: $${selectedOption.strike}`);
    console.log(`💰 Precio de mercado: $${selectedOption.last_price}`);
    console.log(`🧮 Modelos probados: ${models.length}`);
    
  } catch (error) {
    console.error('❌ Error en la prueba:', error.message);
    if (error.response) {
      console.error('Respuesta del servidor:', error.response);
    }
  }
}

// Ejecutar la prueba
testModelSelection();
