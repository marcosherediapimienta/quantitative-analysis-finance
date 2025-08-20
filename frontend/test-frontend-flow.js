const API_BASE_URL = 'http://localhost:8000';

async function testFrontendFlow() {
  console.log('🧪 Simulando el flujo exacto del frontend...\n');
  
  try {
    // Simular el estado inicial del hook
    let yahooData = {
      expirations: [],
      optionsChain: [],
      currentPrice: null,
      selectedOption: null
    };
    
    let userInputs = {
      symbol: 'AAPL',
      optionType: 'call',
      expirationDate: '',
      strike: '',
      riskFreeRate: '',
      volatility: '',
      useImpliedVolatility: false,
      marketPrice: '',
      selectedModel: 'black_scholes',
      nSteps: 100,
      nSimulations: 10000,
      seed: null
    };
    
    // Paso 1: Obtener expiraciones
    console.log('📅 Paso 1: Obteniendo expiraciones...');
    const expirationsResponse = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/?symbol=${userInputs.symbol}`);
    
    if (!expirationsResponse.ok) {
      throw new Error(`Error obteniendo expiraciones: HTTP ${expirationsResponse.status}`);
    }
    
    const expirationsData = await expirationsResponse.json();
    console.log(`✅ Expiraciones obtenidas: ${expirationsData.count} fechas disponibles`);
    
    // Simular selección de expiración
    const selectedExpiration = expirationsData.expirations[0];
    userInputs.expirationDate = selectedExpiration.date;
    console.log(`📅 Expiración seleccionada: ${selectedExpiration.formatted} (${selectedExpiration.date})`);
    
    // Paso 2: Obtener opciones (simulando fetchOptionsChain)
    console.log('\n🎯 Paso 2: Obteniendo opciones (simulando fetchOptionsChain)...');
    const optionsResponse = await fetch(`${API_BASE_URL}/api/option-pricing/yahoo-finance/options/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol: userInputs.symbol,
        expiration_date: userInputs.expirationDate,
        option_type: userInputs.optionType
      }),
    });
    
    if (!optionsResponse.ok) {
      throw new Error(`Error obteniendo opciones: HTTP ${optionsResponse.status}`);
    }
    
    const data = await optionsResponse.json();
    console.log(`✅ Respuesta del backend recibida`);
    
    // Simular exactamente lo que hace el hook
    if (data.status === 'success') {
      console.log('🔍 Debug: Datos recibidos del backend:', data);
      console.log('🔍 Debug: Calls disponibles:', data.calls);
      console.log('🔍 Debug: Puts disponibles:', data.puts);
      console.log('🔍 Debug: Tipo de opción solicitado:', userInputs.optionType);
      
      const optionsChain = data.calls || data.puts || [];
      console.log('🔍 Debug: OptionsChain final:', optionsChain);
      console.log('🔍 Debug: Longitud de optionsChain:', optionsChain.length);
      
      // Simular setYahooData
      yahooData = {
        ...yahooData,
        optionsChain: optionsChain,
        currentPrice: data.current_price,
        selectedOption: null
      };
      
      console.log('🔍 Debug: yahooData después de setYahooData:', yahooData);
      
      // Si hay opciones, seleccionar la primera por defecto
      if (data.calls && data.calls.length > 0) {
        console.log('🔍 Debug: Seleccionando primera opción CALL');
        yahooData.selectedOption = data.calls[0];
      } else if (data.puts && data.puts.length > 0) {
        console.log('🔍 Debug: Seleccionando primera opción PUT');
        yahooData.selectedOption = data.puts[0];
      }
      
      console.log('🔍 Debug: Estado final de yahooData:', yahooData);
      
      // Verificar si las opciones están disponibles
      if (yahooData.optionsChain && yahooData.optionsChain.length > 0) {
        console.log('✅ Opciones disponibles en yahooData.optionsChain');
        console.log(`📊 Total de opciones: ${yahooData.optionsChain.length}`);
        
        // Mostrar las primeras opciones
        console.log('\n📊 Primeras opciones en optionsChain:');
        yahooData.optionsChain.slice(0, 5).forEach((option, index) => {
          const isITM = userInputs.optionType === 'call' ? 
            option.strike < yahooData.currentPrice : 
            option.strike > yahooData.currentPrice;
          const isATM = Math.abs(option.strike - yahooData.currentPrice) < 1;
          
          let status = '';
          if (isITM) status = '🟢 ITM';
          else if (isATM) status = '🟡 ATM';
          else status = '🔴 OTM';
          
          console.log(`   ${index + 1}. Strike: $${option.strike} | Precio: $${option.last_price} | ${status}`);
        });
      } else {
        console.log('❌ ERROR: No hay opciones en yahooData.optionsChain');
        console.log('   - optionsChain:', yahooData.optionsChain);
        console.log('   - optionsChain.length:', yahooData.optionsChain?.length);
      }
      
    } else {
      throw new Error(data.message || 'Error obteniendo opciones');
    }
    
    console.log('\n🎉 Simulación del flujo del frontend completada!');
    
  } catch (error) {
    console.error('❌ Error en la simulación:', error.message);
  }
}

// Ejecutar la simulación
testFrontendFlow();
