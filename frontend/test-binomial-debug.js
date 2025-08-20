const API_BASE_URL = 'http://localhost:8000';

async function testBinomialDebug() {
  console.log('🧪 Debuggeando modelo Binomial...\n');
  
  try {
    // Paso 1: Obtener expiraciones para AAPL
    console.log('📅 Paso 1: Obteniendo expiraciones para AAPL...');
    const expirationsResponse = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/?symbol=AAPL`);
    
    if (!expirationsResponse.ok) {
      throw new Error(`Error obteniendo expiraciones: HTTP ${expirationsResponse.status}`);
    }
    
    const expirationsData = await expirationsResponse.json();
    console.log(`✅ Expiraciones obtenidas: ${expirationsData.count} fechas disponibles`);
    
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
    
    // Paso 3: Crear opción en el backend
    console.log('\n🧮 Paso 3: Creando opción para Binomial...');
    
    const optionData = {
      name: 'AAPL CALL Test Binomial Debug',
      type: 'call',
      style: 'european',
      spot: optionsData.current_price.toFixed(6),
      strike: selectedOption.strike.toFixed(6),
      maturity: (selectedExpiration.days_to_expiry / 365).toFixed(6),
      volatility: (selectedOption.implied_volatility || 0.25).toFixed(6),
      rate: '0.050000',
    };
    
    console.log('📤 Datos de la opción:', optionData);
    
    const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(optionData),
    });
    
    if (!createResponse.ok) {
      const errorData = await createResponse.json();
      console.error('❌ Error creando opción:', errorData);
      return;
    }
    
    const createdOption = await createResponse.json();
    console.log(`✅ Opción creada con ID: ${createdOption.id}`);
    
    // Paso 4: Probar diferentes configuraciones del modelo Binomial
    console.log('\n💰 Paso 4: Probando diferentes configuraciones del modelo Binomial...');
    
    const binomialConfigs = [
      { n_steps: 50, description: 'Pocos pasos (50)' },
      { n_steps: 100, description: 'Pasos estándar (100)' },
      { n_steps: 200, description: 'Muchos pasos (200)' },
      { n_steps: 500, description: 'Muy precisos (500)' }
    ];
    
    for (const config of binomialConfigs) {
      console.log(`\n🧮 Probando: ${config.description}...`);
      
      const pricingData = {
        model: 'binomial',
        calculate_greeks: true,
        n_steps: config.n_steps
      };
      
      console.log('📤 Parámetros:', pricingData);
      
      try {
        const pricingResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/${createdOption.id}/calculate_price/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(pricingData),
        });
        
        if (!pricingResponse.ok) {
          const errorData = await pricingResponse.json();
          console.error(`❌ Error con ${config.n_steps} pasos:`, errorData);
          
          // Si es error 500, mostrar más detalles
          if (pricingResponse.status === 500) {
            console.error(`🔍 Error interno del servidor con ${config.n_steps} pasos`);
            console.error(`   - Status: ${pricingResponse.status}`);
            console.error(`   - Error data:`, errorData);
          }
        } else {
          const pricingResult = await pricingResponse.json();
          console.log(`✅ ${config.description} - Precio: $${pricingResult.calculated_price.toFixed(4)}`);
          console.log(`   ⏱️  Tiempo: ${pricingResult.calculation_time.toFixed(3)}s`);
          
          if (pricingResult.greeks) {
            console.log(`   📊 Delta: ${parseFloat(pricingResult.greeks.delta).toFixed(4)}`);
          }
        }
      } catch (error) {
        console.error(`❌ Excepción con ${config.n_steps} pasos:`, error.message);
      }
    }
    
    // Resumen final
    console.log('\n🎉 ¡Debug del modelo Binomial completado!');
    console.log(`📈 Símbolo: AAPL`);
    console.log(`📅 Expiración: ${selectedExpiration.formatted}`);
    console.log(`🎯 Strike: $${selectedOption.strike}`);
    console.log(`💰 Precio de mercado: $${selectedOption.last_price}`);
    
  } catch (error) {
    console.error('❌ Error en la prueba:', error.message);
    if (error.response) {
      console.error('Respuesta del servidor:', error.response);
    }
  }
}

// Ejecutar la prueba
testBinomialDebug();
