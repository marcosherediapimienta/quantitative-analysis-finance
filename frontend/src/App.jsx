import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Home,
  TrendingUp,
  BarChart3,
  Calculator,
  Download,
  Settings,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
  TrendingDown,
  Activity,
  ArrowRight,
  ArrowLeft,
  Play,
  Target,
  Calendar,
  DollarSign,
  Percent,
  Search
} from "lucide-react";
import { useInteractiveOptionsAnalysis } from "./hooks/useInteractiveOptionsAnalysis";
import SensitivityChart from "./components/SensitivityChart";

// Componentes de UI reutilizables
const Card = ({ className = "", children }) => (
  <div className={`rounded-2xl border border-zinc-800 bg-zinc-900/80 shadow-2xl backdrop-blur-sm text-zinc-100 ${className}`}>
    {children}
  </div>
);

const CardContent = ({ className = "", children }) => (
  <div className={`p-6 ${className}`}>{children}</div>
);

const CardHeader = ({ className = "", children }) => (
  <div className={`p-6 pb-0 ${className}`}>{children}</div>
);

const CardTitle = ({ className = "", children }) => (
  <h3 className={`text-xl font-semibold text-zinc-200 ${className}`}>{children}</h3>
);

const Button = ({ className = "", variant = "default", size = "default", children, ...props }) => {
  const variants = {
    default: "bg-blue-600 hover:bg-blue-700 text-white",
    secondary: "bg-zinc-700 hover:bg-zinc-600 text-zinc-100",
    success: "bg-green-600 hover:bg-green-700 text-white",
    danger: "bg-red-600 hover:bg-red-700 text-white",
    ghost: "bg-transparent hover:bg-zinc-800/60 text-zinc-200",
  };
  
  const sizes = {
    sm: "px-3 py-1.5 text-xs",
    default: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base",
  };
  
  return (
    <button
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-xl font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400/40 ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};

const Input = ({ className = "", label, ...props }) => (
  <div className="space-y-2">
    {label && <label className="text-sm font-medium text-zinc-300">{label}</label>}
  <input
      className={`w-full rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-400/40 ${className}`}
    {...props}
  />
  </div>
);

const Select = ({ className = "", label, children, ...props }) => (
  <div className="space-y-2">
    {label && <label className="text-sm font-medium text-zinc-300">{label}</label>}
  <select
      className={`w-full rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-400/40 ${className}`}
    {...props}
  >
    {children}
  </select>
  </div>
);

const Badge = ({ className = "", color = "blue", children }) => {
  const colorMap = {
    blue: "bg-blue-500/20 text-blue-300 border-blue-500/30",
    green: "bg-green-500/20 text-green-300 border-green-500/30",
    yellow: "bg-yellow-500/20 text-yellow-300 border-yellow-500/30",
    red: "bg-red-500/20 text-red-300 border-red-500/30",
    purple: "bg-purple-500/20 text-purple-300 border-purple-500/30",
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${colorMap[color] || colorMap.blue} ${className}`}>
      {children}
    </span>
  );
};

const LoadingSpinner = ({ size = "md" }) => {
  const sizeMap = { sm: "w-4 h-4", md: "w-6 h-6", lg: "w-8 h-8" };
  return (
    <div className={`animate-spin rounded-full border-2 border-zinc-600 border-t-blue-500 ${sizeMap[size]}`} />
  );
};

// Error Boundary para capturar errores en componentes
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('🚨 Error Boundary capturó un error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 text-white flex items-center justify-center">
          <div className="text-center space-y-4 p-8">
            <div className="w-20 h-20 bg-red-500/20 rounded-2xl flex items-center justify-center mx-auto">
              <AlertCircle className="w-10 h-10 text-red-400" />
            </div>
            <h1 className="text-2xl font-bold text-white">Error en la Aplicación</h1>
            <p className="text-zinc-400 max-w-md">
              Ha ocurrido un error inesperado. La aplicación se ha detenido para prevenir más problemas.
            </p>
            <div className="space-y-2 text-sm text-zinc-500">
              <p>Error: {this.state.error?.message || 'Desconocido'}</p>
              <p>Componente: {this.state.error?.componentStack?.split('\n')[1] || 'N/A'}</p>
            </div>
            <Button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="px-6 py-3 bg-red-600 hover:bg-red-700"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Recargar Aplicación
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Componente principal de la aplicación
export default function App() {
  const [activeTab, setActiveTab] = useState("home");
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

  const tabs = [
    { id: "home", label: "Inicio", icon: <Home size={20} /> },
    { id: "options", label: "Análisis de Opciones", icon: <TrendingUp size={20} /> },
    { id: "portfolio", label: "Portfolio", icon: <BarChart3 size={20} /> },
    { id: "settings", label: "Configuración", icon: <Settings size={20} /> },
  ];

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <span className="text-white text-sm font-black">GA</span>
            </div>
            <span className="text-lg font-bold tracking-wide text-zinc-100">GalaAnalytics</span>
            <Badge color="purple">Pro</Badge>
          </div>
          
          {/* Navegación principal */}
          <nav className="hidden md:flex items-center gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-blue-600/20 text-blue-300 border border-blue-500/30"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Estado del backend */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              backendStatus === 'connected-ok' ? 'bg-green-500' :
              backendStatus === 'error' ? 'bg-red-500' : 'bg-yellow-500'
            }`} />
            <span className="text-xs text-zinc-400 hidden sm:inline">
              {backendStatus === 'connected-ok' ? 'Backend OK' :
               backendStatus === 'error' ? 'Backend Error' : 'Conectando...'}
            </span>
          </div>
        </div>
      </header>

      {/* Navegación móvil */}
      <nav className="md:hidden border-b border-zinc-800 bg-zinc-900/50">
        <div className="flex overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-shrink-0 flex items-center gap-2 px-4 py-3 transition-all duration-200 ${
                activeTab === tab.id
                  ? "text-blue-400 border-b-2 border-blue-400"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Contenido principal */}
      <main className="mx-auto max-w-7xl px-4 py-6">
        <AnimatePresence mode="wait">
          {activeTab === "home" && <HomeTab key="home" backendStatus={backendStatus} />}
          {activeTab === "options" && <OptionsAnalysisTab key="options" />}
          {activeTab === "portfolio" && <PortfolioTab key="portfolio" />}
          {activeTab === "settings" && <SettingsTab key="settings" />}
        </AnimatePresence>
      </main>
    </div>
    </ErrorBoundary>
  );
}

// Tab de inicio
function HomeTab({ backendStatus }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
          Análisis Cuantitativo de Opciones
        </h1>
        <p className="text-xl text-zinc-300 max-w-3xl mx-auto">
          Plataforma avanzada para el análisis, pricing y gestión de opciones financieras usando modelos matemáticos de vanguardia.
        </p>
      </div>

      {/* Estado del sistema */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="text-center">
            <div className="w-12 h-12 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-6 h-6 text-green-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Frontend</h3>
            <p className="text-green-400 text-sm">✅ Funcionando correctamente</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="text-center">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4 ${
              backendStatus === 'connected-ok' ? 'bg-green-500/20' : 'bg-red-500/20'
            }`}>
              {backendStatus === 'connected-ok' ? (
                <CheckCircle className="w-6 h-6 text-green-400" />
              ) : (
                <AlertCircle className="w-6 h-6 text-red-400" />
              )}
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Backend</h3>
            <p className={`text-sm ${
              backendStatus === 'connected-ok' ? 'text-green-400' : 'text-red-400'
            }`}>
              {backendStatus === 'connected-ok' ? '✅ Conectado' : '❌ Error de conexión'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="text-center">
            <div className="w-12 h-12 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <TrendingUp className="w-6 h-6 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Yahoo Finance</h3>
            <p className="text-blue-400 text-sm">📊 API disponible</p>
          </CardContent>
        </Card>
      </div>

      {/* Características principales */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { icon: <Calculator className="w-8 h-8" />, title: "Pricing Avanzado", desc: "Black-Scholes, Binomial, Monte Carlo" },
          { icon: <TrendingUp className="w-8 h-8" />, title: "Análisis de Griegas", desc: "Delta, Gamma, Vega, Theta, Rho" },
          { icon: <BarChart3 className="w-8 h-8" />, title: "Sensibilidad", desc: "Análisis de escenarios y stress testing" },
          { icon: <Download className="w-8 h-8" />, title: "Datos en Tiempo Real", desc: "Yahoo Finance API integrada" },
        ].map((feature, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card className="hover:scale-105 transition-transform duration-200 cursor-pointer">
              <CardContent className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <div className="text-blue-400">{feature.icon}</div>
                  </div>
                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-zinc-400 text-sm">{feature.desc}</p>
                </CardContent>
              </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// Tab de análisis de opciones interactivo
function OptionsAnalysisTab() {
  // Función de prueba de conectividad local para este componente
  const testBackendConnectivity = async () => {
    try {
      console.log('🔍 Probando conectividad del backend desde OptionsAnalysisTab...');
      
      // Probar endpoint de health
      const healthResponse = await fetch('http://localhost:8000/api/option-pricing/health/');
      console.log('✅ Health check:', healthResponse.status, healthResponse.ok);
      
      // Probar endpoint de opciones
      const optionsResponse = await fetch('http://localhost:8000/api/option-pricing/options/');
      console.log('✅ Options endpoint:', optionsResponse.status, optionsResponse.ok);
      
      // Probar endpoint de análisis de sensibilidad (con un ID dummy)
      const sensitivityTestResponse = await fetch('http://localhost:8000/api/option-pricing/options/999999/sensitivity_analysis/', {
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
  };
  const {
    currentStep,
    loading,
    error,
    userInputs,
    yahooData,
    analysisResults,
    sensitivityParams,
    sensitivityResults,
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
    setYahooData
  } = useInteractiveOptionsAnalysis();

  // Manejo de errores global para prevenir pantalla en blanco
  const [globalError, setGlobalError] = useState(null);
  
  // Capturar errores no manejados
  useEffect(() => {
    const handleError = (event) => {
      console.error('🚨 Error global capturado:', event.error);
      setGlobalError(event.error?.message || 'Error inesperado en la aplicación');
    };

    const handleUnhandledRejection = (event) => {
      console.error('🚨 Promesa rechazada no manejada:', event.reason);
      setGlobalError(event.reason?.message || 'Error en promesa no manejada');
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  // Limpiar error global cuando se resetea el flujo
  useEffect(() => {
    if (currentStep === 0) {
      setGlobalError(null);
    }
  }, [currentStep]);

  // Validación automática para opciones americanas
  useEffect(() => {
    if (userInputs.optionStyle === 'american' && userInputs.selectedModel === 'black_scholes') {
      updateUserInput('selectedModel', 'binomial');
    }
  }, [userInputs.optionStyle, userInputs.selectedModel, updateUserInput]);

  // Calcular rangos automáticos cuando cambie el tipo de sensibilidad
  useEffect(() => {
    calculateSensitivityRanges();
  }, [sensitivityParams.sensitivityType, calculateSensitivityRanges]);

  const steps = [
    { id: 0, title: "Símbolo", description: "Ingresa el símbolo de la acción" },
    { id: 1, title: "Expiraciones", description: "Selecciona la fecha de expiración" },
    { id: 2, title: "Resultados", description: "Análisis completo de la opción" },
    { id: 3, title: "Sensibilidad", description: "Análisis de sensibilidad y escenarios" }
  ];

  const handleSymbolSubmit = async (e) => {
    e.preventDefault();
    if (userInputs.symbol.trim()) {
      try {
        await fetchExpirations(userInputs.symbol.trim());
      } catch (err) {
        console.error('Error:', err);
      }
    }
  };

  const handleExpirationSelect = async (expiration) => {
    updateUserInput('expirationDate', expiration.date);
    try {
      await fetchOptionsChain(userInputs.symbol, expiration.date, userInputs.optionType);
      nextStep(); // Usar nextStep en lugar de setCurrentStep(2)
    } catch (err) {
      console.error('Error:', err);
    }
  };

  const handleCalculatePrice = async () => {
    try {
      console.log('🚀 === INICIANDO CÁLCULO DESDE INTERFAZ ===');
      console.log('📋 Estado antes del cálculo:', {
        currentStep,
        hasSelectedOption: !!yahooData.selectedOption,
        hasRiskFreeRate: !!userInputs.riskFreeRate,
        hasCurrentPrice: !!yahooData.currentPrice,
        expirationDate: userInputs.expirationDate
      });
      
      // Verificar que todos los campos necesarios estén llenos
      if (!yahooData.selectedOption) {
        console.error('❌ No hay opción seleccionada');
        setError('Debes seleccionar una opción antes de calcular el precio');
        return;
      }
      
      if (!userInputs.riskFreeRate) {
        console.error('❌ No hay tasa libre de riesgo');
        setError('Debes ingresar la tasa libre de riesgo');
        return;
      }
      
      if (!yahooData.currentPrice || yahooData.currentPrice <= 0) {
        console.error('❌ Precio actual no válido:', yahooData.currentPrice);
        setError('El precio actual del subyacente no es válido. Intenta seleccionar otra expiración.');
        return;
      }
      
      if (!userInputs.expirationDate) {
        console.error('❌ No hay fecha de expiración');
        setError('Debes seleccionar una fecha de expiración');
        return;
      }
      
      console.log('✅ Validaciones de interfaz pasadas, llamando a calculateOptionPrice...');
      
      const result = await calculateOptionPrice();
      
      if (result) {
        console.log('✅ Cálculo completado exitosamente desde interfaz');
      } else {
        console.log('⚠️ Cálculo no completado (probablemente error manejado)');
      }
      
    } catch (err) {
      console.error('🚨 Error en handleCalculatePrice:', err);
      setError(`Error inesperado: ${err.message}`);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Análisis Interactivo de Opciones</h1>
        <div className="flex items-center gap-2">
          {userInputs.optionStyle === 'american' && (
            <Badge color="orange" className="flex items-center gap-1">
              <span>🇺🇸</span>
              Americana
            </Badge>
          )}
          <Badge color="blue">Beta</Badge>
          <Button variant="ghost" onClick={resetFlow} size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Reiniciar
          </Button>
          <Button 
            variant="ghost" 
            onClick={() => {
              console.log('🔍 Estado actual de la aplicación:');
              console.log('Current Step:', currentStep);
              console.log('User Inputs:', userInputs);
              console.log('Yahoo Data:', yahooData);
              console.log('Analysis Results:', analysisResults);
              console.log('Loading:', loading);
              console.log('Error:', error);
            }} 
            size="sm"
            className="text-xs"
          >
            🐛 Debug
          </Button>
        </div>
      </div>

      {/* Error Global */}
      {globalError && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-500/20 border border-red-500/30 rounded-xl p-4 text-red-300"
        >
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5" />
            <h3 className="font-semibold">Error Crítico de la Aplicación</h3>
          </div>
          <p className="text-sm">{globalError}</p>
          <Button
            onClick={() => setGlobalError(null)}
            variant="ghost"
            size="sm"
            className="mt-2 text-red-300 hover:text-red-100"
          >
            Cerrar
          </Button>
        </motion.div>
      )}

      {/* Indicador de pasos */}
      <div className="flex items-center justify-center">
        <div className="flex items-center space-x-4">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                currentStep >= step.id 
                  ? 'border-blue-500 bg-blue-500 text-white' 
                  : 'border-zinc-600 text-zinc-400'
              }`}>
                {currentStep > step.id ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  <span className="text-sm font-semibold">{step.id + 1}</span>
                )}
          </div>
              {index < steps.length - 1 && (
                <div className={`w-16 h-0.5 mx-2 ${
                  currentStep > step.id ? 'bg-blue-500' : 'bg-zinc-600'
                }`} />
              )}
          </div>
          ))}
          </div>
          </div>

      {/* Mensaje de error */}
      {error && (
        <Card className="border-red-500/30 bg-red-500/10">
          <CardContent>
            <div className="flex items-center gap-2 text-red-400">
              <AlertCircle className="w-5 h-5" />
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Contenido del paso actual */}
      <AnimatePresence mode="wait">
        {currentStep === 0 && (
          <motion.div
            key="step-0"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <span className="text-blue-400 font-bold">1</span>
                  </div>
                  Selecciona el Símbolo de la Acción
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSymbolSubmit} className="space-y-6">
                  <div className="text-center space-y-4">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mx-auto">
                      <Target className="w-10 h-10 text-blue-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-white">¿Qué acción quieres analizar?</h3>
                    <p className="text-zinc-400">Ingresa el símbolo de la empresa (ej: AAPL, MSFT, GOOGL)</p>
                  </div>
                  
                  <div className="max-w-md mx-auto space-y-6">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-zinc-300">Símbolo de la Acción</label>
                      <Input
                        value={userInputs.symbol}
                        onChange={(e) => updateUserInput('symbol', e.target.value.toUpperCase())}
                        placeholder="AAPL"
                        className="text-center text-lg h-12"
                      />
                      <p className="text-xs text-zinc-500 text-center">El símbolo debe estar en mayúsculas</p>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-zinc-300">Tipo de Opción</label>
                      <Select
                        value={userInputs.optionType}
                        onChange={(e) => updateUserInput('optionType', e.target.value)}
                        className="h-12"
                      >
                        <option value="">Selecciona el tipo de opción</option>
                        <option value="call">📈 Call (Opción de Compra)</option>
                        <option value="put">📉 Put (Opción de Venta)</option>
                      </Select>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-zinc-300">Estilo de Opción</label>
                      <Select
                        value={userInputs.optionStyle}
                        onChange={(e) => updateUserInput('optionStyle', e.target.value)}
                        className="h-12"
                      >
                        <option value="american">🇺🇸 Americana (Ejercicio Anticipado)</option>
                        <option value="european">🇪🇺 Europea (Solo en Vencimiento)</option>
                      </Select>
                      <p className="text-xs text-zinc-500 text-center">
                        {userInputs.optionStyle === 'american' 
                          ? 'Puede ejercerse en cualquier momento hasta vencimiento' 
                          : 'Solo puede ejercerse en la fecha de vencimiento'}
                      </p>
                    </div>
                    
                    {/* Información educativa sobre estilos de opciones */}
                    <div className="mt-6 p-4 bg-gradient-to-r from-blue-900/20 to-purple-900/20 rounded-xl border border-blue-500/30">
                      <h4 className="text-sm font-semibold text-blue-300 mb-3 flex items-center gap-2">
                        <Calculator className="w-4 h-4" />
                        Diferencias Clave entre Estilos de Opciones
                      </h4>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Opciones Americanas */}
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-orange-400 text-lg">🇺🇸</span>
                            <span className="text-sm font-medium text-orange-300">Opciones Americanas</span>
                          </div>
                          <ul className="space-y-1 text-xs text-blue-200">
                            <li>• <strong>Ejercicio:</strong> En cualquier momento hasta vencimiento</li>
                            <li>• <strong>Flexibilidad:</strong> Máxima flexibilidad estratégica</li>
                            <li>• <strong>Precio:</strong> ≥ Opción europea equivalente</li>
                            <li>• <strong>Modelo:</strong> Binomial o Monte Carlo</li>
                            <li>• <strong>Uso:</strong> Acciones, ETFs, índices</li>
                          </ul>
                        </div>
                        
                        {/* Opciones Europeas */}
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-blue-400 text-lg">🇪🇺</span>
                            <span className="text-sm font-medium text-blue-300">Opciones Europeas</span>
                          </div>
                          <ul className="space-y-1 text-xs text-blue-200">
                            <li>• <strong>Ejercicio:</strong> Solo en fecha de vencimiento</li>
                            <li>• <strong>Flexibilidad:</strong> Limitada al vencimiento</li>
                            <li>• <strong>Precio:</strong> ≤ Opción americana equivalente</li>
                            <li>• <strong>Modelo:</strong> Black-Scholes, Binomial, MC</li>
                            <li>• <strong>Uso:</strong> Índices, futuros, FX</li>
                          </ul>
                        </div>
                      </div>
                      
                      <div className="mt-3 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                        <p className="text-xs text-blue-200">
                          <strong>Nota:</strong> En la práctica, la mayoría de opciones sobre acciones individuales son americanas, 
                          mientras que las opciones sobre índices suelen ser europeas. La elección del estilo afecta 
                          significativamente la estrategia de trading y el modelo de pricing utilizado.
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <Button
                      type="submit"
                      disabled={!userInputs.symbol.trim() || !userInputs.optionType || loading}
                      className="px-10 py-4 text-lg font-medium bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                    >
                      {loading ? (
                        <>
                          <LoadingSpinner size="sm" className="mr-2" />
                          Buscando Opciones...
                        </>
                      ) : (
                        <>
                          <Search className="w-5 h-5 mr-2" />
                          Buscar Opciones Disponibles
                          <ArrowRight className="w-5 h-5 ml-2" />
                        </>
                      )}
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {currentStep === 1 && (
          <motion.div
            key="step-1"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <span className="text-blue-400 font-bold">2</span>
                  </div>
                  Selecciona la Fecha de Expiración
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center space-y-4 mb-6">
                  <div className="w-20 h-20 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mx-auto">
                    <Calendar className="w-10 h-10 text-blue-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white">Elige la fecha de expiración</h3>
                  <p className="text-zinc-400">Selecciona cuándo vence la opción que quieres analizar</p>
                </div>

                {/* Lista de Expiraciones Mejorada */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-zinc-300">Fechas de Expiración Disponibles</label>
                    <div className="text-xs text-zinc-500 bg-zinc-800/50 px-2 py-1 rounded">
                      {yahooData.expirations?.length || 0} fechas disponibles
                    </div>
                  </div>
                  
                  <div className="max-h-80 overflow-y-auto bg-zinc-800/50 rounded-lg p-4">
                    {yahooData.expirations && yahooData.expirations.length > 0 ? (
                      <div className="grid gap-3">
                        {yahooData.expirations.map((expiration, index) => {
                          // Calcular si es próxima, media o lejana
                          const days = expiration.days_to_expiry;
                          let timeCategory = '';
                          let timeColor = '';
                          let timeBg = '';
                          
                          if (days <= 7) {
                            timeCategory = 'Próxima';
                            timeColor = 'text-red-400';
                            timeBg = 'bg-red-500/20';
                          } else if (days <= 30) {
                            timeCategory = 'Corta';
                            timeColor = 'text-orange-400';
                            timeBg = 'bg-orange-500/20';
                          } else if (days <= 90) {
                            timeCategory = 'Media';
                            timeColor = 'text-yellow-400';
                            timeBg = 'bg-yellow-500/20';
                          } else {
                            timeCategory = 'Larga';
                            timeColor = 'text-green-400';
                            timeBg = 'bg-green-500/20';
                          }
                          
                          return (
                            <div
                              key={index}
                              onClick={() => handleExpirationSelect(expiration)}
                              className="group p-4 rounded-xl cursor-pointer transition-all duration-200 bg-zinc-700/30 hover:bg-zinc-600/40 hover:border-blue-500/50 border border-transparent hover:shadow-lg hover:shadow-blue-500/10"
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex-1">
                                  <div className="flex items-center gap-3 mb-2">
                                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${timeBg} ${timeColor}`}>
                                      {timeCategory}
                                    </div>
                                    <div className="text-sm text-zinc-500">
                                      {expiration.date}
                                    </div>
                                  </div>
                                  <div className="font-semibold text-white text-lg group-hover:text-blue-300 transition-colors">
                                    {expiration.formatted}
                                  </div>
                                </div>
                                
                                <div className="text-right ml-4">
                                  <div className={`text-2xl font-bold ${timeColor} mb-1`}>
                                    {expiration.days_to_expiry}
                                  </div>
                                  <div className="text-xs text-zinc-500 uppercase tracking-wide">
                                    días
                                  </div>
                                  <div className="text-xs text-zinc-400 mt-1">
                                    hasta vencimiento
                                  </div>
                                </div>
                                
                                <div className="ml-4 opacity-0 group-hover:opacity-100 transition-opacity">
                                  <ArrowRight className="w-5 h-5 text-blue-400" />
                                </div>
                              </div>
                              
                              {/* Información adicional */}
                              <div className="mt-3 pt-3 border-t border-zinc-600/30 opacity-0 group-hover:opacity-100 transition-opacity">
                                <div className="flex items-center justify-between text-xs text-zinc-400">
                                  <span>Click para seleccionar</span>
                                  <span className="text-blue-400">→ Continuar</span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="text-center text-zinc-500 py-12">
                        <Calendar className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p className="text-lg">No hay fechas de expiración disponibles</p>
                        <p className="text-sm">Intenta con otro símbolo o verifica la conexión</p>
                      </div>
                    )}
                  </div>
                  
                  {/* Leyenda de categorías */}
                  <div className="flex items-center justify-center gap-4 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-500/20 rounded-full border border-red-500/50"></div>
                      <span className="text-zinc-400">Próxima (≤7 días)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-orange-500/20 rounded-full border border-orange-500/50"></div>
                      <span className="text-zinc-400">Corta (≤30 días)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-yellow-500/20 rounded-full border border-yellow-500/50"></div>
                      <span className="text-zinc-400">Media (≤90 días)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500/20 rounded-full border border-green-500/50"></div>
                      <span className="text-zinc-400">Larga (&gt;90 días)</span>
                    </div>
                  </div>
                </div>

                {/* Botones de Navegación */}
                <div className="flex justify-between items-center pt-4 border-t border-zinc-700/50">
                  <Button onClick={prevStep} variant="ghost" className="hover:bg-zinc-700/50">
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    ← Anterior
                  </Button>
                  
                  <div className="text-xs text-zinc-500 bg-zinc-800/50 px-3 py-1 rounded-full">
                    Selecciona una fecha para continuar
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {currentStep === 2 && (
          <motion.div
            key="step-2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
        <Card>
          <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <span className="text-blue-400 font-bold">3</span>
                  </div>
                  Configuración y Resultados
                </CardTitle>
          </CardHeader>
          <CardContent>
                <div className="text-center space-y-4 mb-6">
                  <div className="w-20 h-20 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mx-auto">
                    <DollarSign className="w-10 h-10 text-blue-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white">Configuración y Resultados</h3>
                  <p className="text-zinc-400">Configura los parámetros del modelo y obtén el análisis completo</p>
                </div>

                {/* Opciones Disponibles Mejoradas */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-zinc-300">Strikes Disponibles</label>
                    <div className="text-xs text-zinc-500 bg-zinc-800/50 px-2 py-1 rounded">
                      {yahooData.optionsChain?.length || 0} opciones disponibles
                    </div>
                  </div>
                  
                  <div className="max-h-80 overflow-y-auto bg-zinc-800/50 rounded-lg p-4">
                    {yahooData.optionsChain && yahooData.optionsChain.length > 0 ? (
                      <div className="grid gap-3">
                        {yahooData.optionsChain.map((option, index) => {
                          const isSelected = yahooData.selectedOption?.strike === option.strike;
                          const isITM = userInputs.optionType === 'call' ? 
                            option.strike < yahooData.currentPrice : 
                            option.strike > yahooData.currentPrice;
                          const isATM = Math.abs(option.strike - yahooData.currentPrice) < 1;
                          
                          let statusColor = '';
                          let statusBg = '';
                          let statusText = '';
                          
                          if (isATM) {
                            statusColor = 'text-yellow-400';
                            statusBg = 'bg-yellow-500/20';
                            statusText = 'ATM';
                          } else if (isITM) {
                            statusColor = 'text-green-400';
                            statusBg = 'bg-green-500/20';
                            statusText = 'ITM';
                          } else {
                            statusColor = 'text-red-400';
                            statusBg = 'bg-red-500/20';
                            statusText = 'OTM';
                          }
                          
                          return (
                            <div
                              key={index}
                              onClick={() => selectOption(option)}
                              className={`group p-4 rounded-xl cursor-pointer transition-all duration-200 ${
                                isSelected 
                                  ? 'bg-blue-600/30 border border-blue-500/50 shadow-lg shadow-blue-500/20' 
                                  : 'bg-zinc-700/30 hover:bg-zinc-600/40 hover:border-blue-500/30 border border-transparent'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex-1">
                                  <div className="flex items-center gap-3 mb-2">
                                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${statusBg} ${statusColor}`}>
                                      {statusText}
                                    </div>
                                    <div className="text-sm text-zinc-500">
                                      Vol: {option.volume || 0}
                                    </div>
                                  </div>
                                  <div className="font-semibold text-white text-lg group-hover:text-blue-300 transition-colors">
                                    Strike: ${option.strike}
                                  </div>
                                  <div className="text-sm text-zinc-400 mt-1">
                                    Precio: ${option.last_price?.toFixed(2) || '0.00'} | 
                                    Vol. Impl.: {((option.implied_volatility || 0) * 100).toFixed(1)}%
                                  </div>
                                </div>
                                
                                <div className="text-right ml-4">
                                  <div className={`text-2xl font-bold ${statusColor} mb-1`}>
                                    ${option.last_price?.toFixed(2) || '0.00'}
                                  </div>
                                  <div className="text-xs text-zinc-500 uppercase tracking-wide">
                                    Premio
                                  </div>
                                  <div className="text-xs text-zinc-400 mt-1">
                                    Bid: ${option.bid?.toFixed(2) || '0.00'} | Ask: ${option.ask?.toFixed(2) || '0.00'}
                                  </div>
                                </div>
                                
                                <div className="ml-4 opacity-0 group-hover:opacity-100 transition-opacity">
                                  {isSelected ? (
                                    <CheckCircle className="w-6 h-6 text-blue-400" />
                                  ) : (
                                    <ArrowRight className="w-5 h-5 text-blue-400" />
                                  )}
                                </div>
                              </div>
                              
                              {/* Información adicional */}
                              <div className="mt-3 pt-3 border-t border-zinc-600/30 opacity-0 group-hover:opacity-100 transition-opacity">
                                <div className="flex items-center justify-between text-xs text-zinc-400">
                                  <span>
                                    {isATM ? 'En el dinero' : isITM ? 'En el dinero' : 'Fuera del dinero'}
                                  </span>
                                  <span className="text-blue-400">
                                    {isSelected ? '✓ Seleccionado' : '→ Click para seleccionar'}
                                  </span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="text-center text-zinc-500 py-12">
                        <DollarSign className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p className="text-lg">No hay strikes disponibles</p>
                        <p className="text-sm">Verifica la fecha de expiración seleccionada</p>
                      </div>
                    )}
                  </div>
                  
                  {/* Leyenda de estados */}
                  <div className="flex items-center justify-center gap-4 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500/20 rounded-full border border-green-500/50"></div>
                      <span className="text-zinc-400">ITM (En el dinero)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-yellow-500/20 rounded-full border border-yellow-500/50"></div>
                      <span className="text-zinc-400">ATM (En el dinero)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-500/20 rounded-full border border-red-500/50"></div>
                      <span className="text-zinc-400">OTM (Fuera del dinero)</span>
                    </div>
                  </div>
                </div>

                {/* Opción Seleccionada Mejorada */}
                {yahooData.selectedOption && (
                  <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-6 border border-blue-500/30">
                    <div className="flex items-center gap-3 mb-4">
                      <CheckCircle className="w-6 h-6 text-green-400" />
                      <h4 className="text-xl font-semibold text-white">Opción Seleccionada</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Información Principal */}
                      <div className="space-y-4">
                        <div className="p-4 bg-zinc-800/50 rounded-lg">
                          <div className="text-sm text-zinc-400 mb-1">Strike Seleccionado</div>
                          <div className="text-3xl font-bold text-blue-400">${yahooData.selectedOption.strike}</div>
                        </div>
                        
                        <div className="p-4 bg-zinc-800/50 rounded-lg">
                          <div className="text-sm text-zinc-400 mb-1">Premio de la Opción</div>
                          <div className="text-3xl font-bold text-green-400">${yahooData.selectedOption.last_price?.toFixed(2) || '0.00'}</div>
                        </div>
                        
                        <div className="p-4 bg-zinc-800/50 rounded-lg">
                          <div className="text-sm text-zinc-400 mb-1">Estilo de Opción</div>
                          <div className={`text-2xl font-bold ${
                            userInputs.optionStyle === 'american' ? 'text-orange-400' : 'text-blue-400'
                          }`}>
                            {userInputs.optionStyle === 'american' ? '🇺🇸 Americana' : '🇪🇺 Europea'}
                          </div>
                          <div className="text-xs text-zinc-400 mt-1">
                            {userInputs.optionStyle === 'american' 
                              ? 'Ejercicio anticipado permitido' 
                              : 'Solo ejercicio en vencimiento'}
                          </div>
                        </div>
                      </div>
                      
                      {/* Detalles Adicionales */}
                      <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-zinc-800/30 rounded-lg">
                          <span className="text-zinc-400">Volatilidad Implícita</span>
                          <span className="font-semibold text-white">{((yahooData.selectedOption.implied_volatility || 0) * 100).toFixed(2)}%</span>
                        </div>
                        
                        <div className="flex justify-between items-center p-3 bg-zinc-800/30 rounded-lg">
                          <span className="text-zinc-400">Volumen</span>
                          <span className="font-semibold text-white">{yahooData.selectedOption.volume || 0}</span>
                        </div>
                        
                        <div className="flex justify-between items-center p-3 bg-zinc-800/30 rounded-lg">
                          <span className="text-zinc-400">Bid / Ask</span>
                          <span className="font-semibold text-white">
                            ${yahooData.selectedOption.bid?.toFixed(2) || '0.00'} / ${yahooData.selectedOption.ask?.toFixed(2) || '0.00'}
                          </span>
                        </div>
                        
                        <div className="flex justify-between items-center p-3 bg-zinc-800/30 rounded-lg">
                          <span className="text-zinc-400">Open Interest</span>
                          <span className="font-semibold text-white">{yahooData.selectedOption.open_interest || 0}</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Comparación con Precio Actual */}
                    <div className="mt-4 p-4 bg-gradient-to-r from-green-900/20 to-blue-900/20 rounded-lg border border-green-500/30">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-sm text-green-300">Precio Actual del Subyacente</div>
                          <div className="text-2xl font-bold text-green-400">${yahooData.currentPrice}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm text-blue-300">Diferencia vs Strike</div>
                          <div className={`text-xl font-bold ${
                            yahooData.currentPrice > yahooData.selectedOption.strike ? 'text-green-400' : 'text-red-400'
                          }`}>
                            ${Math.abs(yahooData.currentPrice - yahooData.selectedOption.strike).toFixed(2)}
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Información específica para opciones americanas */}
                    {userInputs.optionStyle === 'american' && (
                      <div className="mt-4 p-4 bg-gradient-to-r from-orange-900/20 to-red-900/20 rounded-lg border border-orange-500/30">
                        <div className="flex items-center gap-3 mb-3">
                          <div className="w-8 h-8 bg-orange-500/20 rounded-full flex items-center justify-center">
                            <span className="text-orange-400 text-lg">🇺🇸</span>
                          </div>
                          <h4 className="text-lg font-semibold text-orange-300">Opción Americana</h4>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                              <span className="text-green-300">Ventaja del ejercicio anticipado</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                              <span className="text-blue-300">Mayor flexibilidad estratégica</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                              <span className="text-purple-300">Protección contra movimientos adversos</span>
                            </div>
                          </div>
                          
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                              <span className="text-yellow-300">Precio ≥ opción europea equivalente</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                              <span className="text-red-300">Decisión de ejercicio más compleja</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-pink-400 rounded-full"></div>
                              <span className="text-pink-300">Modelo binomial recomendado</span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="mt-3 p-3 bg-orange-500/10 rounded-lg border border-orange-500/20">
                          <p className="text-xs text-orange-200">
                            <strong>Nota:</strong> Las opciones americanas pueden ejercerse en cualquier momento hasta vencimiento, 
                            lo que las hace más valiosas que las europeas equivalentes. El modelo binomial es especialmente 
                            adecuado para calcular su precio considerando todas las oportunidades de ejercicio anticipado.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Configuración de Parámetros Mejorada */}
                {yahooData.selectedOption && (
                  <div className="space-y-6 mt-6">
                    <div className="flex items-center gap-3">
                      <Settings className="w-6 h-6 text-blue-400" />
                      <h4 className="text-xl font-semibold text-white">Configuración del Modelo</h4>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Parámetros Financieros */}
                      <div className="space-y-6">
                        <div className="p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50">
                          <h5 className="font-semibold text-white mb-4 flex items-center gap-2">
                            <Percent className="w-4 h-4" />
                            Parámetros Financieros
                          </h5>
                          
                          <div className="space-y-4">
                            <div className="space-y-2">
                              <label className="text-sm font-medium text-zinc-300">Tasa Libre de Riesgo</label>
                              <div className="relative">
                                <Input
                                  type="number"
                                  step="0.01"
                                  min="0.0"
                                  max="99.9"
                                  value={userInputs.riskFreeRate}
                                  onChange={(e) => updateUserInput('riskFreeRate', e.target.value)}
                                  placeholder="5.0"
                                  className="pr-8"
                                />
                                <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-zinc-400 text-sm">%</span>
                              </div>
                              <div className="text-xs text-zinc-500">
                                Rango válido: 0.0% - 99.9% (máximo 8 dígitos totales)
                              </div>
                              {userInputs.riskFreeRate && parseFloat(userInputs.riskFreeRate) > 99.9 && (
                                <div className="text-xs text-red-400">
                                  ⚠️ La tasa excede el límite máximo del backend
                                </div>
                              )}
                            </div>
                            
                            <div className="space-y-3">
                              <label className="text-sm font-medium text-zinc-300">Configuración de Volatilidad</label>
                              
                              <div className="space-y-3">
                                <div
                                  onClick={() => updateUserInput('useImpliedVolatility', true)}
                                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                                    userInputs.useImpliedVolatility 
                                      ? 'bg-blue-600/20 border border-blue-500/50' 
                                      : 'bg-zinc-700/30 hover:bg-zinc-600/40 border border-transparent'
                                  }`}
                                >
                                  <div className="flex items-center gap-3">
                                    <div className={`w-4 h-4 rounded-full border-2 ${
                                      userInputs.useImpliedVolatility 
                                        ? 'border-blue-500 bg-blue-500' 
                                        : 'border-zinc-400'
                                    }`}>
                                      {userInputs.useImpliedVolatility && (
                                        <div className="w-2 h-2 bg-white rounded-full m-0.5"></div>
                                      )}
                                    </div>
                                    <div>
                                      <div className="text-sm font-medium text-white">Volatilidad Implícita</div>
                                      <div className="text-xs text-zinc-400">Usar la volatilidad del mercado</div>
                                    </div>
                                  </div>
                                  {userInputs.useImpliedVolatility && (
                                    <div className="mt-2 ml-7 text-sm text-blue-300">
                                      Actual: {((yahooData.selectedOption.implied_volatility || 0) * 100).toFixed(2)}%
                                    </div>
                                  )}
                                </div>
                                
                                <div
                                  onClick={() => updateUserInput('useImpliedVolatility', false)}
                                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                                    !userInputs.useImpliedVolatility 
                                      ? 'bg-blue-600/20 border border-blue-500/50' 
                                      : 'bg-zinc-700/30 hover:bg-zinc-600/40 border border-transparent'
                                  }`}
                                >
                                  <div className="flex items-center gap-3">
                                    <div className={`w-4 h-4 rounded-full border-2 ${
                                      !userInputs.useImpliedVolatility 
                                        ? 'border-blue-500 bg-blue-500' 
                                        : 'border-zinc-400'
                                    }`}>
                                      {!userInputs.useImpliedVolatility && (
                                        <div className="w-2 h-2 bg-white rounded-full m-0.5"></div>
                                      )}
                                    </div>
                                    <div>
                                      <div className="text-sm font-medium text-white">Volatilidad Personalizada</div>
                                      <div className="text-xs text-zinc-400">Usar mi propia estimación</div>
                                    </div>
                                  </div>
                                </div>
                              </div>
                              
                              {!userInputs.useImpliedVolatility && (
                                <div className="space-y-2">
                                  <div className="relative">
                                    <Input
                                      type="number"
                                      step="0.1"
                                      min="0.1"
                                      max="99.9"
                                      value={userInputs.volatility}
                                      onChange={(e) => updateUserInput('volatility', e.target.value)}
                                      placeholder="25.0"
                                      className="pr-8"
                                    />
                                    <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-zinc-400 text-sm">%</span>
                                  </div>
                                  <div className="text-xs text-zinc-500">
                                    Rango válido: 0.1% - 99.9% (máximo 8 dígitos totales)
                                  </div>
                                  {userInputs.volatility && parseFloat(userInputs.volatility) > 99.9 && (
                                    <div className="text-xs text-red-400">
                                      ⚠️ La volatilidad excede el límite máximo del backend
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Selección de Modelo */}
                      <div className="space-y-6">
                        <div className="p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50">
                          <h5 className="font-semibold text-white mb-4 flex items-center gap-2">
                            <Calculator className="w-4 h-4" />
                            Modelo de Pricing
                          </h5>
                          
                          {/* Recomendación para opciones americanas */}
                          {userInputs.optionStyle === 'american' && (
                            <div className="mb-4 p-3 bg-orange-500/10 rounded-lg border border-orange-500/30">
                              <div className="flex items-center gap-2 mb-2">
                                <span className="text-orange-400">💡</span>
                                <span className="text-sm font-medium text-orange-300">Recomendación para Opciones Americanas</span>
                              </div>
                              <p className="text-xs text-orange-200">
                                Para opciones americanas, el modelo <strong>Binomial</strong> es el más adecuado ya que considera 
                                todas las oportunidades de ejercicio anticipado. Black-Scholes solo es válido para opciones europeas.
                              </p>
                            </div>
                          )}
                          
                          <div className="space-y-3">
                            {[
                              {
                                value: 'black_scholes',
                                name: 'Black-Scholes',
                                description: 'Modelo analítico clásico',
                                icon: '📈',
                                speed: 'Muy rápido',
                                disabled: userInputs.optionStyle === 'american'
                              },
                              {
                                value: 'binomial',
                                name: 'Binomial',
                                description: 'Modelo de árbol binomial',
                                icon: '🌳',
                                speed: 'Rápido',
                                disabled: false
                              },
                              {
                                value: 'monte_carlo',
                                name: 'Monte Carlo',
                                description: 'Simulación estocástica',
                                icon: '🎲',
                                speed: 'Más lento',
                                disabled: false
                              }
                            ].map((model) => (
                              <div
                                key={model.value}
                                onClick={() => !model.disabled && updateUserInput('selectedModel', model.value)}
                                className={`p-3 rounded-lg transition-all ${
                                  model.disabled 
                                    ? 'bg-zinc-600/20 border border-zinc-500/30 cursor-not-allowed opacity-50' :
                                  userInputs.selectedModel === model.value
                                    ? 'bg-blue-600/20 border border-blue-500/50 cursor-pointer' 
                                    : 'bg-zinc-700/30 hover:bg-zinc-600/40 border border-transparent cursor-pointer'
                                }`}
                              >
                                <div className="flex items-center gap-3">
                                  <span className="text-xl">{model.icon}</span>
                                  <div className="flex-1">
                                    <div className="text-sm font-medium text-white">{model.name}</div>
                                    <div className="text-xs text-zinc-400">{model.description}</div>
                                    {model.disabled && (
                                      <div className="text-xs text-red-400 mt-1">
                                        No disponible para opciones americanas
                                      </div>
                                    )}
                                  </div>
                                  <div className="text-right">
                                    <div className={`text-xs px-2 py-1 rounded ${
                                      userInputs.selectedModel === model.value
                                        ? 'bg-blue-500/20 text-blue-300'
                                        : 'bg-zinc-600/50 text-zinc-400'
                                    }`}>
                                      {model.speed}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                          
                          {/* Parámetros específicos del modelo */}
                          {userInputs.selectedModel === 'binomial' && (
                            <div className="mt-4 p-3 bg-blue-900/20 rounded-lg border border-blue-500/30">
                              <label className="text-sm font-medium text-blue-300 mb-2 block">Número de Pasos</label>
                              <Input
                                type="number"
                                min="10"
                                max="10000"
                                value={userInputs.nSteps}
                                onChange={(e) => updateUserInput('nSteps', e.target.value)}
                                placeholder="100"
                              />
                              <p className="text-xs text-blue-400 mt-1">Más pasos = mayor precisión pero más lento</p>
                            </div>
                          )}
                          
                          {userInputs.selectedModel === 'monte_carlo' && (
                            <div className="mt-4 space-y-3">
                              <div className="p-3 bg-purple-900/20 rounded-lg border border-purple-500/30">
                                <label className="text-sm font-medium text-purple-300 mb-2 block">Número de Simulaciones</label>
                                <Input
                                  type="number"
                                  min="1000"
                                  max="100000"
                                  value={userInputs.nSimulations}
                                  onChange={(e) => updateUserInput('nSimulations', e.target.value)}
                                  placeholder="10000"
                                />
                                <p className="text-xs text-purple-400 mt-1">Más simulaciones = mayor precisión pero más lento</p>
                              </div>
                              
                              <div className="p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                                <label className="text-sm font-medium text-yellow-300 mb-2 block">Semilla (opcional)</label>
                                <Input
                                  type="number"
                                  value={userInputs.seed || ''}
                                  onChange={(e) => updateUserInput('seed', e.target.value || null)}
                                  placeholder="Dejar vacío para aleatorio"
                                />
                                <p className="text-xs text-yellow-400 mt-1">Para resultados reproducibles</p>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <div className="p-4 bg-blue-500/10 rounded-xl border border-blue-500/20">
                          <h4 className="text-sm font-semibold text-blue-300 mb-2">Resumen de la Opción</h4>
                          <div className="space-y-1 text-xs text-blue-200">
                            <div><strong>Símbolo:</strong> {userInputs.symbol}</div>
                            <div><strong>Tipo:</strong> {userInputs.optionType.toUpperCase()}</div>
                            <div><strong>Estilo:</strong> {userInputs.optionStyle === 'american' ? '🇺🇸 Americana' : '🇪🇺 Europea'}</div>
                            <div><strong>Expiracion:</strong> {userInputs.expirationDate}</div>
                            <div><strong>Strike:</strong> ${yahooData.selectedOption?.strike}</div>
                            <div><strong>Spot:</strong> ${yahooData.currentPrice}</div>
                            <div><strong>Tasa:</strong> {userInputs.riskFreeRate}%</div>
                            <div><strong>Volatilidad:</strong> {userInputs.useImpliedVolatility ? 
                              `${(yahooData.selectedOption?.implied_volatility * 100).toFixed(2)}%` : 
                              `${userInputs.volatility}%`}</div>
                            <div><strong>Modelo:</strong> {userInputs.selectedModel === 'black_scholes' ? 'Black-Scholes' : userInputs.selectedModel === 'binomial' ? 'Binomial' : 'Monte Carlo'}</div>
                            {userInputs.selectedModel === 'binomial' && (
                              <div><strong>Pasos:</strong> {userInputs.nSteps}</div>
                            )}
                            {userInputs.selectedModel === 'monte_carlo' && (
                              <div><strong>Simulaciones:</strong> {userInputs.nSimulations}</div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Botones de navegación */}
                    <div className="flex items-center justify-center gap-4">
                      <Button variant="ghost" onClick={prevStep}>
                        <ArrowLeft className="w-4 h-4 mr-2" />
                        Anterior
                      </Button>
                      
                      <Button
                        onClick={handleCalculatePrice}
                        disabled={loading || !userInputs.riskFreeRate || (userInputs.optionStyle === 'american' && userInputs.selectedModel === 'black_scholes')}
                        className="px-8 py-3"
                      >
                        {loading ? (
                          <>
                            <LoadingSpinner size="sm" className="mr-2" />
                            Calculando...
                          </>
                        ) : (
                          <>
                            <Calculator className="w-4 h-4 mr-2" />
                            Calcular Precio
                          </>
                        )}
                      </Button>
                      
                      {/* Mensaje de validación para opciones americanas */}
                      {userInputs.optionStyle === 'american' && userInputs.selectedModel === 'black_scholes' && (
                        <div className="text-center">
                          <p className="text-sm text-red-400">
                            ⚠️ Las opciones americanas no pueden usar el modelo Black-Scholes. 
                            Selecciona Binomial o Monte Carlo.
                          </p>
                        </div>
                      )}
                    </div>
                    
                    {/* Mostrar botón para análisis de sensibilidad después de calcular precio */}
                    {analysisResults.calculatedPrice && (
                      <div className="text-center mt-6">
                        <Button
                          onClick={nextStep}
                          className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                        >
                          <Activity className="w-5 h-5 mr-2" />
                          Continuar con Análisis de Sensibilidad
                          <ArrowRight className="w-5 h-5 ml-2" />
                        </Button>
                        <p className="text-xs text-zinc-500 mt-2">
                          Analiza cómo cambia el precio ante variaciones en los parámetros del mercado
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}

        {currentStep === 3 && (
          <motion.div
            key="step-3"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <span className="text-blue-400 font-bold">4</span>
                  </div>
                  Análisis de Sensibilidad y Escenarios
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center space-y-4 mb-6">
                  <div className="w-20 h-20 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl flex items-center justify-center mx-auto">
                    <Activity className="w-10 h-10 text-purple-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white">Análisis de Sensibilidad</h3>
                  <p className="text-zinc-400">Analiza cómo cambia el precio de la opción ante variaciones en los parámetros del mercado</p>
                </div>

                {/* Configuración del Análisis de Sensibilidad */}
                <div className="space-y-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Parámetros de Sensibilidad */}
                    <div className="p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50">
                      <h5 className="font-semibold text-white mb-4 flex items-center gap-2">
                        <Activity className="w-4 h-4" />
                        Configuración del Análisis
                      </h5>
                      
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-zinc-300">Tipo de Sensibilidad</label>
                          <Select
                            value={sensitivityParams.sensitivityType}
                            onChange={(e) => updateSensitivityParams('sensitivityType', e.target.value)}
                          >
                            <option value="spot">📈 Precio del Subyacente (±10%)</option>
                            <option value="volatility">📊 Volatilidad (10% - 30%)</option>
                            <option value="rate">🏦 Tasa de Interés (1% - 5%)</option>
                            <option value="time">⏰ Tiempo al Vencimiento (50% - 150%)</option>
                          </Select>
                          <p className="text-xs text-zinc-500">
                            {sensitivityParams.sensitivityType === 'spot' && 'Analiza cambios del ±10% en el precio de la acción'}
                            {sensitivityParams.sensitivityType === 'volatility' && 'Analiza volatilidad entre 10% y 30%'}
                            {sensitivityParams.sensitivityType === 'rate' && 'Analiza tasas de interés entre 1% y 5%'}
                            {sensitivityParams.sensitivityType === 'time' && 'Analiza entre 50% y 150% del tiempo restante'}
                          </p>
                        </div>
                        
                        {/* Información sobre el modelo */}
                        <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                          <p className="text-xs text-blue-200">
                            <strong>💡 Nota:</strong> El análisis usará automáticamente el modelo <strong>
                            {userInputs.selectedModel === 'black_scholes' ? 'Black-Scholes' : 
                              userInputs.selectedModel === 'binomial' ? 'Binomial' : 'Monte Carlo'}
                            </strong> seleccionado en el paso 3.
                          </p>
                        </div>
                      </div>
                    </div>
                    
                    {/* Información del Análisis */}
                    <div className="space-y-4">
                      <div className="p-4 bg-purple-500/10 rounded-xl border border-purple-500/30">
                        <h5 className="font-semibold text-purple-300 mb-3">¿Qué es el Análisis de Sensibilidad?</h5>
                        <div className="space-y-2 text-sm text-purple-200">
                          <p>El análisis de sensibilidad te permite entender cómo cambia el precio de tu opción ante variaciones en los parámetros del mercado.</p>
                          <p><strong>Beneficios:</strong></p>
                          <ul className="list-disc list-inside space-y-1 ml-2">
                            <li>Identificar riesgos principales</li>
                            <li>Optimizar estrategias de trading</li>
                            <li>Prepararse para diferentes escenarios</li>
                            <li>Mejorar la gestión de riesgo</li>
                          </ul>
                        </div>
                      </div>
                      
                      <div className="p-4 bg-blue-500/10 rounded-xl border border-blue-500/30">
                        <h5 className="font-semibold text-blue-300 mb-2">Rangos Simplificados</h5>
                        <div className="text-xs text-blue-200 space-y-1">
                          <div><strong>Spot:</strong> ±10% del precio actual</div>
                          <div><strong>Volatilidad:</strong> 10% - 30%</div>
                          <div><strong>Tasa:</strong> 1% - 5%</div>
                          <div><strong>Tiempo:</strong> 50% - 150% del tiempo restante</div>
                        </div>
                      </div>
                      
                      <div className="p-4 bg-green-500/10 rounded-xl border border-green-500/30">
                        <h5 className="font-semibold text-green-300 mb-2">Estado Actual</h5>
                        <div className="text-xs text-green-200">
                          <div><strong>Opción:</strong> {userInputs.symbol} {userInputs.optionType.toUpperCase()}</div>
                          <div><strong>Estilo:</strong> {userInputs.optionStyle === 'american' ? '🇺🇸 Americana' : '🇪🇺 Europea'}</div>
                          <div><strong>Strike:</strong> ${yahooData.selectedOption?.strike}</div>
                          <div><strong>Spot:</strong> ${yahooData.currentPrice}</div>
                          <div><strong>Volatilidad:</strong> {userInputs.useImpliedVolatility ? 
                            `${(yahooData.selectedOption?.implied_volatility * 100).toFixed(2)}%` : 
                            `${userInputs.volatility}%`}</div>
                          <div><strong>Tiempo Vencimiento:</strong> {userInputs.expirationDate ? Math.ceil((new Date(userInputs.expirationDate) - new Date()) / (1000 * 60 * 60 * 24)) : 0} días</div>
                          <div><strong>Tipo de Interés:</strong> {userInputs.riskFreeRate}%</div>
                          <div><strong>Modelo:</strong> {userInputs.selectedModel === 'black_scholes' ? 'Black-Scholes' : 
                            userInputs.selectedModel === 'binomial' ? 'Binomial' : 'Monte Carlo'}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Botones de Análisis */}
                  <div className="text-center space-y-4">
                    <Button
                      onClick={performSensitivityAnalysis}
                      disabled={sensitivityResults.loading}
                      className="px-10 py-4 text-lg font-medium bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                    >
                      {sensitivityResults.loading ? (
                        <>
                          <LoadingSpinner size="sm" className="mr-2" />
                          Analizando Sensibilidad...
                        </>
                      ) : (
                        <>
                          <Activity className="w-5 h-5 mr-2" />
                          Realizar Análisis de Sensibilidad
                          <ArrowRight className="w-5 h-5 ml-2" />
                        </>
                      )}
                    </Button>
                    
                    {/* Botón de prueba de conectividad */}
                    <div className="flex justify-center">
                      <Button
                        onClick={testBackendConnectivity}
                        variant="secondary"
                        size="sm"
                        className="text-xs"
                      >
                        🔍 Probar Conectividad del Backend
                      </Button>
                    </div>
                  </div>
                  
                  {/* Resultados del Análisis */}
                  {sensitivityResults.error && (
                    <div className="p-4 bg-red-500/10 rounded-xl border border-red-500/30">
                      <div className="flex items-center gap-2 text-red-400">
                        <AlertCircle className="w-5 h-5" />
                        <span>{sensitivityResults.error}</span>
                      </div>
                    </div>
                  )}
                  
                  {sensitivityResults.data && (
                    <div className="space-y-4">
                      <div className="p-4 bg-green-500/10 rounded-xl border border-green-500/30">
                        <h5 className="font-semibold text-green-300 mb-3">Resultados del Análisis</h5>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                          <div className="text-center p-3 bg-green-500/20 rounded-lg">
                            <div className="text-lg font-bold text-green-400">
                              {sensitivityResults.data.data_points?.length || 0}
                            </div>
                            <div className="text-green-200">Puntos Analizados</div>
                          </div>
                          
                          <div className="text-center p-3 bg-blue-500/20 rounded-lg">
                            <div className="text-lg font-bold text-blue-400">
                              {sensitivityResults.data.calculation_time?.toFixed(2) || '0.00'}s
                            </div>
                            <div className="text-blue-200">Tiempo de Cálculo</div>
                          </div>
                          
                          <div className="text-center p-3 bg-purple-500/20 rounded-lg">
                            <div className="text-lg font-bold text-purple-400">
                              {sensitivityResults.data.model_used || 'N/A'}
                            </div>
                            <div className="text-purple-200">Modelo Utilizado</div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Gráfico de Sensibilidad */}
                      <div className="p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50">
                        <h5 className="font-semibold text-white mb-4">Gráfico de Sensibilidad</h5>
                        <SensitivityChart 
                          data={sensitivityResults.data}
                          sensitivityType={sensitivityParams.sensitivityType}
                          basePrice={analysisResults.calculatedPrice}
                        />
                      </div>
                      
                      {/* Tabla de Datos */}
                      <div className="p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50">
                        <h5 className="font-semibold text-white mb-4">Datos del Análisis</h5>
                        <div className="max-h-64 overflow-y-auto">
                          <table className="w-full text-sm">
                            <thead className="bg-zinc-700/50">
                              <tr>
                                <th className="text-left p-2 text-zinc-300">Parámetro</th>
                                <th className="text-right p-2 text-zinc-300">Precio Opción</th>
                                <th className="text-right p-2 text-zinc-300">Cambio</th>
                              </tr>
                            </thead>
                            <tbody>
                              {sensitivityResults.data.data_points?.map((point, index) => {
                                const basePrice = analysisResults.calculatedPrice || 0;
                                const change = point.option_price - basePrice;
                                const changePercent = basePrice > 0 ? (change / basePrice) * 100 : 0;
                                
                                return (
                                  <tr key={index} className="border-t border-zinc-700/30">
                                    <td className="p-2 text-zinc-200">
                                      {sensitivityParams.sensitivityType === 'spot' && '$'}
                                      {sensitivityParams.sensitivityType === 'volatility' && ''}
                                      {sensitivityParams.sensitivityType === 'rate' && ''}
                                      {sensitivityParams.sensitivityType === 'time' && ''}
                                      {point.parameter_value.toFixed(4)}
                                      {sensitivityParams.sensitivityType === 'volatility' && '%'}
                                      {sensitivityParams.sensitivityType === 'rate' && '%'}
                                      {sensitivityParams.sensitivityType === 'time' && ' años'}
                                    </td>
                                    <td className="p-2 text-right text-zinc-200">
                                      ${point.option_price.toFixed(4)}
                                    </td>
                                    <td className={`p-2 text-right ${
                                      change > 0 ? 'text-green-400' : change < 0 ? 'text-red-400' : 'text-zinc-400'
                                    }`}>
                                      {change > 0 ? '+' : ''}{change.toFixed(4)} ({changePercent > 0 ? '+' : ''}{changePercent.toFixed(2)}%)
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Botones de Navegación */}
                  <div className="flex items-center justify-center gap-4 pt-4 border-t border-zinc-700/50">
                    <Button onClick={prevStep} variant="ghost">
                      <ArrowLeft className="w-4 h-4 mr-2" />
                      Anterior
                    </Button>
                    
                    <Button onClick={resetFlow} variant="secondary">
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Nuevo Análisis
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Tab de portfolio
function PortfolioTab() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <h1 className="text-3xl font-bold text-white">Portfolio de Opciones</h1>
      
      <Card>
        <CardContent>
          <div className="text-center text-zinc-500 py-12">
            <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg">Portfolio en desarrollo</p>
            <p className="text-sm">Próximamente podrás gestionar tus posiciones de opciones</p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Tab de configuración
function SettingsTab() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <h1 className="text-3xl font-bold text-white">Configuración</h1>
      
      <Card>
        <CardContent>
          <div className="text-center text-zinc-500 py-12">
            <Settings className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg">Configuración en desarrollo</p>
            <p className="text-sm">Próximamente podrás personalizar tu experiencia</p>
      </div>
        </CardContent>
      </Card>
    </motion.div>
  );
} 