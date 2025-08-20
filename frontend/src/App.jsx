import React, { useEffect, useState } from "react";

export default function App() {
  const [tab, setTab] = useState("home");
  const [backendStatus, setBackendStatus] = useState("checking");

  useEffect(() => {
    document.documentElement.classList.add("dark");
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/option-pricing/health/');
      if (response.ok) {
        const data = await response.json();
        setBackendStatus(`connected-${data.status}`);
      } else {
        setBackendStatus('error');
      }
    } catch (error) {
      setBackendStatus('error');
      console.error('Error conectando con el backend:', error);
    }
  };

  const getBackendStatusText = () => {
    switch (backendStatus) {
      case 'connected-ok':
        return '✅ Conectado al Backend';
      case 'error':
        return '❌ Error de Conexión';
      case 'checking':
        return '⏳ Verificando Conexión...';
      default:
        return '❓ Estado Desconocido';
    }
  };

  const getBackendStatusColor = () => {
    switch (backendStatus) {
      case 'connected-ok':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
      case 'checking':
        return 'text-yellow-400';
      default:
        return 'text-zinc-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 text-white">
      <header className="sticky top-0 z-40 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md">
        <div className="mx-auto max-w-md md:max-w-2xl lg:max-w-5xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-6 w-6 rounded-md bg-blue-500/20 ring-1 ring-blue-400/40 flex items-center justify-center">
              <span className="text-blue-300 text-xs font-black">GA</span>
            </div>
            <span className="text-sm font-semibold tracking-wide text-zinc-100">GalaAnalytics</span>
          </div>
        </div>
      </header>
      
      <main className="mx-auto max-w-md md:max-w-2xl lg:max-w-5xl pb-24 px-4">
        <div className="pt-6">
          <h1 className="text-3xl font-bold text-white mb-4">
            ¡Bienvenido a GalaAnalytics!
          </h1>
          <p className="text-zinc-300 mb-6">
            Tu centro de análisis financiero está funcionando correctamente.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="p-6 bg-zinc-900/80 rounded-2xl border border-zinc-800">
              <h2 className="text-xl font-semibold text-white mb-2">Estado del Frontend</h2>
              <p className="text-green-400">✅ Funcionando correctamente</p>
            </div>
            
            <div className="p-6 bg-zinc-900/80 rounded-2xl border border-zinc-800">
              <h2 className="text-xl font-semibold text-white mb-2">Próximos Pasos</h2>
              <p className="text-zinc-300">Configurar conexión con el backend</p>
            </div>
          </div>

          {/* Estado de la conexión con el backend */}
          <div className="p-6 bg-zinc-900/80 rounded-2xl border border-zinc-800">
            <h2 className="text-xl font-semibold text-white mb-4">Estado de la Conexión Backend</h2>
            <div className="flex items-center gap-3 mb-4">
              <span className={`text-lg font-medium ${getBackendStatusColor()}`}>
                {getBackendStatusText()}
              </span>
            </div>
            
            {backendStatus === 'connected-ok' && (
              <div className="space-y-2">
                <p className="text-green-400 text-sm">✅ El backend está respondiendo correctamente</p>
                <p className="text-zinc-300 text-sm">🌐 URL: http://localhost:8000</p>
                <p className="text-zinc-300 text-sm">🔗 Endpoint de prueba: /api/option-pricing/health/</p>
              </div>
            )}
            
            {backendStatus === 'error' && (
              <div className="space-y-2">
                <p className="text-red-400 text-sm">❌ No se pudo conectar con el backend</p>
                <p className="text-zinc-300 text-sm">Verifica que el backend esté ejecutándose en el puerto 8000</p>
                <button 
                  onClick={checkBackendConnection}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white text-sm transition-colors"
                >
                  Reintentar Conexión
                </button>
              </div>
            )}
            
            {backendStatus === 'checking' && (
              <div className="space-y-2">
                <p className="text-yellow-400 text-sm">⏳ Verificando conexión con el backend...</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}