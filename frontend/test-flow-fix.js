const API_BASE_URL = 'http://localhost:8000';

async function testFlowFix() {
  console.log('🧪 Probando el flujo corregido...\n');
  
  try {
    // Paso 1: Obtener expiraciones para AAPL
    console.log('📅 Paso 1: Obteniendo expiraciones para AAPL...');
    const expirationsResponse = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/?symbol=AAPL`);
    
    if (!expirationsResponse.ok) {
      throw new Error(`Error obteniendo expiraciones: HTTP ${expirationsResponse.status}`);
    }
    
    const expirationsData = await expirationsResponse.json();
    console.log(`✅ Expiraciones obtenidas: ${expirationsData.count} fechas disponibles`);
    
    // Seleccionar la primera expiración
    const selectedExpiration = expirationsData.expirations[0];
    console.log(`📅 Expiración seleccionada: ${selectedExpiration.formatted} (${selectedExpiration.date})`);
    
    // Paso 2: Obtener opciones CALL para la expiración seleccionada
    console.log('\n🎯 Paso 2: Obteniendo opciones CALL...');
    const optionsResponse = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol: 'AAPL',
        expiration_date: selectedExpiration.date,
        option_type: 'call'
      }),
    });
    
    if (!optionsResponse.ok) {
      throw new Error(`Error obteniendo opciones: HTTP ${optionsResponse.status}`);
    }
    
    const optionsData = await optionsResponse.json();
    console.log(`✅ Opciones obtenidas: ${optionsData.total_calls} calls disponibles`);
    
    // Debug: mostrar la estructura completa de la respuesta
    console.log('\n🔍 Estructura de la respuesta del backend:');
    console.log('   - status:', optionsData.status);
    console.log('   - symbol:', optionsData.symbol);
    console.log('   - expiration_date:', optionsData.expiration_date);
    console.log('   - current_price:', optionsData.current_price);
    console.log('   - total_calls:', optionsData.total_calls);
    console.log('   - total_puts:', optionsData.total_puts);
    console.log('   - calls array length:', optionsData.calls ? optionsData.calls.length : 'undefined');
    console.log('   - puts array length:', optionsData.puts ? optionsData.puts.length : 'undefined');
    
    if (optionsData.total_calls === 0) {
      throw new Error('No se encontraron opciones CALL');
    }
    
    // Mostrar información de las primeras opciones
    console.log('\n📊 Primeras opciones disponibles:');
    const firstOptions = optionsData.calls.slice(0, 5);
    firstOptions.forEach((option, index) => {
      const isITM = option.strike < optionsData.current_price;
      const isATM = Math.abs(option.strike - optionsData.current_price) < 1;
      const isOTM = option.strike > optionsData.current_price;
      
      let status = '';
      if (isITM) status = '🟢 ITM';
      else if (isATM) status = '🟡 ATM';
      else if (isOTM) status = '🔴 OTM';
      
      console.log(`   ${index + 1}. Strike: $${option.strike} | Precio: $${option.last_price} | ${status}`);
    });
    
    // Verificar el precio actual
    console.log(`\n💰 Precio actual del subyacente: $${optionsData.current_price}`);
    
    if (!optionsData.current_price || optionsData.current_price <= 0) {
      console.error('❌ ERROR: El precio actual no es válido');
      return;
    }
    
    console.log('✅ El flujo está funcionando correctamente!');
    console.log(`📈 Símbolo: AAPL`);
    console.log(`📅 Expiración: ${selectedExpiration.formatted}`);
    console.log(`🎯 Tipo: CALL`);
    console.log(`💰 Precio de mercado: $${optionsData.current_price}`);
    console.log(`📊 Opciones disponibles: ${optionsData.total_calls}`);
    
  } catch (error) {
    console.error('❌ Error en la prueba:', error.message);
  }
}

// Ejecutar la prueba
testFlowFix();
