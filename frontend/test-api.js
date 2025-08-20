// Script de prueba para verificar la API
const API_BASE_URL = 'http://localhost:8000';

async function testAPI() {
  console.log('🧪 Probando API de opciones...');
  
  try {
    // 1. Test health check
    console.log('1. Probando health check...');
    const healthResponse = await fetch(`${API_BASE_URL}/api/option-pricing/health/`);
    if (healthResponse.ok) {
      const healthData = await healthResponse.json();
      console.log('✅ Health check OK:', healthData);
    } else {
      console.log('❌ Health check falló:', healthResponse.status);
    }
    
    // 2. Crear opción
    console.log('\n2. Creando opción...');
    const createResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: 'TEST CALL',
        type: 'call',
        style: 'european',
        spot: '150.00',
        strike: '150.00',
        maturity: '0.25',
        volatility: '0.25',
        rate: '0.05',
      }),
    });
    
    if (createResponse.ok) {
      const option = await createResponse.json();
      console.log('✅ Opción creada:', option);
      
      // 3. Calcular precio
      console.log('\n3. Calculando precio...');
      const pricingResponse = await fetch(`${API_BASE_URL}/api/option-pricing/options/${option.id}/calculate_price/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'black_scholes',
          calculate_greeks: true,
        }),
      });
      
      if (pricingResponse.ok) {
        const pricingData = await pricingResponse.json();
        console.log('✅ Precio calculado:', pricingData);
        console.log('💰 Precio:', pricingData.calculated_price);
        console.log('📊 Griegas:', pricingData.greeks);
      } else {
        const errorText = await pricingResponse.text();
        console.log('❌ Error calculando precio:', pricingResponse.status, errorText);
      }
      
    } else {
      const errorText = await createResponse.text();
      console.log('❌ Error creando opción:', createResponse.status, errorText);
    }
    
  } catch (error) {
    console.error('❌ Error en la prueba:', error);
  }
}

// Ejecutar la prueba
testAPI();
