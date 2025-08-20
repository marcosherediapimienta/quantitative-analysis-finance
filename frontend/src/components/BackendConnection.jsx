import React, { useEffect, useState } from 'react';
import { useServicesHealth, useOptionPricing } from '@/hooks/useApi';
import config from '@/config';

const BackendConnection = () => {
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const { checkAllServices, loading, error, data } = useServicesHealth();
  const { getYahooFinanceTest, getYahooFinanceTicker } = useOptionPricing();
  const [yahooData, setYahooData] = useState(null);
  const [tickerData, setTickerData] = useState(null);

  useEffect(() => {
    checkConnection();
  }, []);

  const checkConnection = async () => {
    try {
      setConnectionStatus('checking');
      const result = await checkAllServices();
      
      if (result.allServices === 'healthy') {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('error');
      }
    } catch (err) {
      setConnectionStatus('error');
      console.error('Error checking connection:', err);
    }
  };

  const testYahooFinance = async () => {
    try {
      const result = await getYahooFinanceTest();
      setYahooData(result);
    } catch (err) {
      console.error('Error testing Yahoo Finance:', err);
    }
  };

  const testTicker = async () => {
    try {
      const result = await getYahooFinanceTicker('AAPL');
      setTickerData(result);
    } catch (err) {
      console.error('Error testing ticker:', err);
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'text-green-600 bg-green-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      case 'checking':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected':
        return '✅ Conectado al Backend';
      case 'error':
        return '❌ Error de Conexión';
      case 'checking':
        return '⏳ Verificando Conexión...';
      default:
        return '❓ Estado Desconocido';
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">
        Conexión Backend-Frontend
      </h2>

      {/* Estado de la conexión */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-700">
          Estado de la Conexión
        </h3>
        <div className={`p-3 rounded-lg ${getStatusColor()}`}>
          <span className="font-medium">{getStatusText()}</span>
        </div>
      </div>

      {/* Información de configuración */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-700">
          Configuración
        </h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium">URL del Backend:</span>
              <p className="text-blue-600 font-mono">{config.API_BASE_URL}</p>
            </div>
            <div>
              <span className="font-medium">Entorno:</span>
              <p className="text-gray-600">{config.NODE_ENV}</p>
            </div>
            <div>
              <span className="font-medium">Timeout:</span>
              <p className="text-gray-600">{config.API_TIMEOUT}ms</p>
            </div>
            <div>
              <span className="font-medium">Modo Debug:</span>
              <p className="text-gray-600">{config.ENABLE_DEBUG_MODE ? 'Activado' : 'Desactivado'}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Botones de prueba */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3 text-gray-700">
          Pruebas de API
        </h3>
        <div className="flex gap-3">
          <button
            onClick={checkConnection}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Verificando...' : 'Verificar Conexión'}
          </button>
          <button
            onClick={testYahooFinance}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            Probar Yahoo Finance
          </button>
          <button
            onClick={testTicker}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
          >
            Probar Ticker (AAPL)
          </button>
        </div>
      </div>

      {/* Resultados de las pruebas */}
      {data && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-gray-700">
            Estado de los Servicios
          </h3>
          <div className="bg-gray-50 p-4 rounded-lg">
            <pre className="text-sm overflow-auto">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {yahooData && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-gray-700">
            Datos de Yahoo Finance
          </h3>
          <div className="bg-gray-50 p-4 rounded-lg">
            <pre className="text-sm overflow-auto">
              {JSON.stringify(yahooData, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {tickerData && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-gray-700">
            Datos del Ticker AAPL
          </h3>
          <div className="bg-gray-50 p-4 rounded-lg">
            <pre className="text-sm overflow-auto">
              {JSON.stringify(tickerData, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {/* Errores */}
      {error && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-red-700">
            Error
          </h3>
          <div className="bg-red-50 p-4 rounded-lg border border-red-200">
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Información adicional */}
      <div className="text-sm text-gray-600">
        <p>
          <strong>Nota:</strong> Este componente verifica la conexión entre el frontend 
          y el backend. Asegúrate de que el servidor de Django esté ejecutándose en 
          el puerto 8000.
        </p>
      </div>
    </div>
  );
};

export default BackendConnection;
