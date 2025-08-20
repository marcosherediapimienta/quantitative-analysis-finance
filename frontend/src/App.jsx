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
  Percent
} from "lucide-react";
import { useInteractiveOptionsAnalysis } from "./hooks/useInteractiveOptionsAnalysis";

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
  const {
    currentStep,
    loading,
    error,
    userInputs,
    yahooData,
    analysisResults,
    fetchExpirations,
    fetchOptionsChain,
    calculateOptionPrice,
    calculateImpliedVolatility,
    updateUserInput,
    selectOption,
    resetFlow,
    nextStep,
    prevStep,
    setYahooData
  } = useInteractiveOptionsAnalysis();

  const steps = [
    { id: 0, title: "Símbolo", description: "Ingresa el símbolo de la acción" },
    { id: 1, title: "Expiraciones", description: "Selecciona la fecha de expiración" },
    { id: 2, title: "Opciones", description: "Elige la opción y parámetros" },
    { id: 3, title: "Configuración", description: "Ajusta parámetros del modelo" },
    { id: 4, title: "Resultados", description: "Análisis completo de la opción" }
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
      await calculateOptionPrice();
    } catch (err) {
      console.error('Error:', err);
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
          <Badge color="blue">Beta</Badge>
          <Button variant="ghost" onClick={resetFlow} size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Reiniciar
          </Button>
        </div>
      </div>

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
                <CardTitle>Paso 1: Selecciona el Símbolo de la Acción</CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSymbolSubmit} className="space-y-6">
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mx-auto">
                      <Target className="w-8 h-8 text-blue-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-white">¿Qué acción quieres analizar?</h3>
                    <p className="text-zinc-400">Ingresa el símbolo (ej: AAPL, MSFT, GOOGL)</p>
                  </div>
                  
                  <div className="max-w-md mx-auto space-y-4">
                    <Input
                      label="Símbolo (Ticker)"
                      value={userInputs.symbol}
                      onChange={(e) => updateUserInput('symbol', e.target.value.toUpperCase())}
                      placeholder="AAPL"
                      className="text-center text-lg"
                    />
                    
                    <Select
                      label="Tipo de Opción"
                      value={userInputs.optionType}
                      onChange={(e) => updateUserInput('optionType', e.target.value)}
                    >
                      <option value="">Selecciona el tipo</option>
                      <option value="call">Call (Compra)</option>
                      <option value="put">Put (Venta)</option>
                    </Select>
                  </div>
                  
                  <div className="text-center">
                    <Button
                      type="submit"
                      disabled={!userInputs.symbol.trim() || !userInputs.optionType || loading}
                      className="px-8 py-3"
                    >
                      {loading ? (
                        <>
                          <LoadingSpinner size="sm" className="mr-2" />
                          Buscando...
                        </>
                      ) : (
                        <>
                          Buscar Opciones
                          <ArrowRight className="w-4 h-4 ml-2" />
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
                <CardTitle>Paso 2: Selecciona la Fecha de Expiración</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mx-auto">
                    <Calendar className="w-8 h-8 text-blue-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-white">Elige la fecha de expiración</h3>
                  <p className="text-zinc-400">Selecciona cuándo vence la opción que quieres analizar</p>
                </div>

                {/* Lista de Expiraciones */}
                <div className="space-y-4">
                  <label className="text-sm font-medium text-zinc-300">Fechas de Expiración Disponibles</label>
                  <div className="max-h-60 overflow-y-auto bg-zinc-800/50 rounded-lg p-4">
                    {yahooData.expirations && yahooData.expirations.length > 0 ? (
                      <div className="space-y-2">
                        {yahooData.expirations.map((expiration, index) => (
                          <div
                            key={index}
                            onClick={() => handleExpirationSelect(expiration)}
                            className="p-3 rounded-lg cursor-pointer transition-all bg-zinc-700/50 hover:bg-zinc-600/50 hover:border-blue-500/50 border border-transparent"
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="font-medium text-white">
                                  {expiration.formatted}
                                </div>
                                <div className="text-sm text-zinc-400">
                                  {expiration.date}
                                </div>
                              </div>
                              <div className="text-right">
                                <div className="text-sm font-medium text-blue-400">
                                  {expiration.days_to_expiry} días
                                </div>
                                <div className="text-xs text-zinc-500">
                                  hasta vencimiento
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center text-zinc-500 py-8">
                        No hay fechas de expiración disponibles
                      </div>
                    )}
                  </div>
                </div>

                {/* Botones de Navegación */}
                <div className="flex justify-between">
                  <Button onClick={prevStep} variant="ghost">
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Anterior
                  </Button>
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
                <CardTitle>Paso 3: Selección de Strike y Configuración</CardTitle>
          </CardHeader>
          <CardContent>
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-white mb-2">Selecciona el Strike y Configura</h3>
                  <p className="text-zinc-400">Elige el strike específico y configura los parámetros del modelo</p>
                </div>

                {/* Opciones Disponibles */}
                <div className="space-y-4">
                  <label className="text-sm font-medium text-zinc-300">Opciones Disponibles</label>
                  {/* Debug temporal */}
                  <div className="text-xs text-yellow-400 bg-yellow-900/20 p-2 rounded">
                    Debug: yahooData.optionsChain = {JSON.stringify(yahooData.optionsChain?.length || 0)} opciones
                  </div>
                  <div className="max-h-60 overflow-y-auto bg-zinc-800/50 rounded-lg p-4">
                    {yahooData.optionsChain && yahooData.optionsChain.length > 0 ? (
                      <div className="space-y-2">
                        {yahooData.optionsChain.map((option, index) => {
                          const isSelected = yahooData.selectedOption?.strike === option.strike;
                          const isITM = userInputs.optionType === 'call' ? 
                            option.strike < yahooData.currentPrice : 
                            option.strike > yahooData.currentPrice;
                          const isATM = Math.abs(option.strike - yahooData.currentPrice) < 1;
                          
                          return (
                            <div
                              key={index}
                                                             onClick={() => setYahooData(prev => ({ ...prev, selectedOption: option }))}
                              className={`p-3 rounded-lg cursor-pointer transition-all ${
                                isSelected 
                                  ? 'bg-blue-600/20 border border-blue-500/50' 
                                  : 'bg-zinc-700/50 hover:bg-zinc-600/50'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-3">
                                  <div className={`w-3 h-3 rounded-full ${
                                    isITM ? 'bg-green-500' : isATM ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}></div>
                                  <div>
                                    <div className="font-medium text-white">
                                      Strike: ${option.strike}
                                    </div>
                                    <div className="text-sm text-zinc-400">
                                      Precio: ${option.last_price} | Vol: {option.volume}
                                    </div>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <div className={`text-sm font-medium ${
                                    isITM ? 'text-green-400' : isATM ? 'text-yellow-400' : 'text-red-400'
                                  }`}>
                                    {isITM ? 'ITM' : isATM ? 'ATM' : 'OTM'}
                                  </div>
                                  <div className="text-xs text-zinc-500">
                                    {isITM ? 'In The Money' : isATM ? 'At The Money' : 'Out of The Money'}
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="text-center text-zinc-500 py-8">
                        No hay opciones disponibles
                      </div>
                    )}
                  </div>
                </div>

                {/* Opción Seleccionada */}
                {yahooData.selectedOption && (
                  <div className="bg-zinc-800/50 rounded-xl p-4">
                    <h4 className="text-lg font-semibold text-white mb-3">Opción Seleccionada</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div><strong>Strike:</strong> ${yahooData.selectedOption.strike}</div>
                      <div><strong>Último Precio:</strong> ${yahooData.selectedOption.last_price}</div>
                      <div><strong>Volatilidad Implícita:</strong> {(yahooData.selectedOption.implied_volatility * 100).toFixed(2)}%</div>
                      <div><strong>Volumen:</strong> {yahooData.selectedOption.volume}</div>
                    </div>
                    <div className="mt-3 p-3 bg-blue-600/20 rounded-lg">
                      <div className="text-sm text-blue-300">Precio Actual del Subyacente:</div>
                      <div className="text-2xl font-bold text-green-400">${yahooData.currentPrice}</div>
                    </div>
                  </div>
                )}

                {/* Configuración del Modelo */}
                <div className="space-y-4">
                  <h4 className="text-lg font-semibold text-white">Configuración del Modelo</h4>
                  
                  {/* Tasa Libre de Riesgo */}
                  <Input
                    label="Tasa Libre de Riesgo (%)"
                    type="number"
                    step="0.01"
                    min="0"
                    max="20"
                    value={userInputs.riskFreeRate}
                    onChange={(e) => updateUserInput('riskFreeRate', e.target.value)}
                    placeholder="4.21"
                  />

                  {/* Volatilidad */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-zinc-300">Volatilidad</label>
                    <div className="space-y-2">
                      <label className="flex items-center space-x-2">
                        <input
                          type="radio"
                          checked={userInputs.useImpliedVolatility}
                          onChange={() => updateUserInput('useImpliedVolatility', true)}
                          className="text-blue-500"
                        />
                        <span className="text-sm text-zinc-300">Usar volatilidad implícita del mercado</span>
                      </label>
                      <label className="flex items-center space-x-2">
                        <input
                          type="radio"
                          checked={!userInputs.useImpliedVolatility}
                          onChange={() => updateUserInput('useImpliedVolatility', false)}
                          className="text-blue-500"
                        />
                        <span className="text-sm text-zinc-300">Usar mi propia volatilidad</span>
                      </label>
                    </div>
                    
                    {!userInputs.useImpliedVolatility && (
                      <Input
                        label="Volatilidad (%)"
                        type="number"
                        step="0.1"
                        min="0.1"
                        max="200"
                        value={userInputs.volatility}
                        onChange={(e) => updateUserInput('volatility', e.target.value)}
                        placeholder="25.0"
                      />
                    )}
                  </div>

                  {/* Selección de Modelo */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-zinc-300">Modelo de Pricing</label>
                    <Select
                      value={userInputs.selectedModel}
                      onChange={(e) => updateUserInput('selectedModel', e.target.value)}
                    >
                      <option value="black_scholes">Black-Scholes (Analítico)</option>
                      <option value="binomial">Binomial (Árbol)</option>
                      <option value="monte_carlo">Monte Carlo (Simulación)</option>
                    </Select>
                    
                    {/* Parámetros específicos del modelo */}
                    {userInputs.selectedModel === 'binomial' && (
                      <Input
                        label="Número de Pasos"
                        type="number"
                        min="10"
                        max="10000"
                        value={userInputs.nSteps}
                        onChange={(e) => updateUserInput('nSteps', e.target.value)}
                        placeholder="100"
                      />
                    )}
                    
                    {userInputs.selectedModel === 'monte_carlo' && (
                      <div className="space-y-2">
                        <Input
                          label="Número de Simulaciones"
                          type="number"
                          min="1000"
                          max="100000"
                          value={userInputs.nSimulations}
                          onChange={(e) => updateUserInput('nSimulations', e.target.value)}
                          placeholder="10000"
                        />
                        <Input
                          label="Semilla (opcional)"
                          type="number"
                          value={userInputs.seed || ''}
                          onChange={(e) => updateUserInput('seed', e.target.value || null)}
                          placeholder="Dejar vacío para aleatorio"
                        />
                      </div>
                    )}
                  </div>
                </div>

                {/* Resumen */}
                {yahooData.selectedOption && (
                  <div className="bg-zinc-800/50 rounded-xl p-4">
                    <h4 className="text-lg font-semibold text-white mb-3">Resumen de la Opción</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div><strong>Símbolo:</strong> {userInputs.symbol}</div>
                      <div><strong>Tipo:</strong> {userInputs.optionType.toUpperCase()}</div>
                      <div><strong>Expiracion:</strong> {userInputs.expirationDate}</div>
                      <div><strong>Strike:</strong> ${yahooData.selectedOption.strike}</div>
                      <div><strong>Spot:</strong> ${yahooData.currentPrice}</div>
                      <div><strong>Tasa:</strong> {userInputs.riskFreeRate}%</div>
                      <div><strong>Volatilidad:</strong> {userInputs.useImpliedVolatility ? 
                        `${(yahooData.selectedOption.implied_volatility * 100).toFixed(2)}%` : 
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
                )}

                {/* Botones de Navegación */}
                <div className="flex justify-between">
                  <Button onClick={prevStep} variant="outline">
                    ← Anterior
                  </Button>
                  <Button 
                    onClick={nextStep}
                    disabled={!yahooData.selectedOption || !userInputs.riskFreeRate || 
                             (!userInputs.useImpliedVolatility && !userInputs.volatility)}
                  >
                    Calcular Precio →
                  </Button>
                </div>
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
                <CardTitle>Paso 4: Configuración y Cálculo</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Información de la opción seleccionada */}
                  {yahooData.selectedOption && (
                    <div className="p-4 bg-zinc-800/30 rounded-xl">
                      <h4 className="text-sm font-semibold text-zinc-300 mb-3">Opción Seleccionada</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-zinc-400">Strike:</span>
                          <div className="text-zinc-200 font-medium">${yahooData.selectedOption.strike}</div>
                        </div>
                        <div>
                          <span className="text-zinc-400">Último Precio:</span>
                          <div className="text-zinc-200 font-medium">${yahooData.selectedOption.last_price}</div>
                        </div>
                        <div>
                          <span className="text-zinc-400">Volatilidad Implícita:</span>
                          <div className="text-zinc-200 font-medium">
                            {yahooData.selectedOption.implied_volatility ? 
                              `${(yahooData.selectedOption.implied_volatility * 100).toFixed(2)}%` : 
                              'N/A'
                            }
                          </div>
                        </div>
                        <div>
                          <span className="text-zinc-400">Volumen:</span>
                          <div className="text-zinc-200 font-medium">{yahooData.selectedOption.volume}</div>
                        </div>
                      </div>
                      
                      {/* Indicador del precio actual */}
                      <div className="mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-blue-300">Precio Actual del Subyacente:</span>
                          <span className={`text-lg font-bold ${yahooData.currentPrice > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            ${yahooData.currentPrice > 0 ? yahooData.currentPrice.toFixed(2) : '0.00'}
                          </span>
                        </div>
                        {yahooData.currentPrice <= 0 && (
                          <div className="text-xs text-red-400 mt-1">
                            ⚠️ Precio no válido. Intenta seleccionar otra expiración.
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Parámetros del usuario */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <Input
                        label="Tasa Libre de Riesgo (%)"
                        type="number"
                        step="0.01"
                        value={userInputs.riskFreeRate}
                        onChange={(e) => updateUserInput('riskFreeRate', e.target.value)}
                        placeholder="5.0"
                      />
                      
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-zinc-300">Volatilidad</label>
                        <div className="space-y-2">
                          <label className="flex items-center space-x-2">
                            <input
                              type="radio"
                              checked={userInputs.useImpliedVolatility}
                              onChange={() => updateUserInput('useImpliedVolatility', true)}
                              className="text-blue-500"
                            />
                            <span className="text-sm text-zinc-300">Usar volatilidad implícita del mercado</span>
                          </label>
                          <label className="flex items-center space-x-2">
                            <input
                              type="radio"
                              checked={!userInputs.useImpliedVolatility}
                              onChange={() => updateUserInput('useImpliedVolatility', false)}
                              className="text-blue-500"
                            />
                            <span className="text-sm text-zinc-300">Usar mi propia volatilidad</span>
                          </label>
                        </div>
                        
                        {!userInputs.useImpliedVolatility && (
                          <Input
                            label="Volatilidad (%)"
                            type="number"
                            step="0.1"
                            value={userInputs.volatility}
                            onChange={(e) => updateUserInput('volatility', e.target.value)}
                            placeholder="25.0"
                          />
                        )}
                      </div>
                      
                      {/* Selección de Modelo */}
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-zinc-300">Modelo de Pricing</label>
                        <Select
                          value={userInputs.selectedModel}
                          onChange={(e) => updateUserInput('selectedModel', e.target.value)}
                        >
                          <option value="black_scholes">Black-Scholes (Analítico)</option>
                          <option value="binomial">Binomial (Árbol)</option>
                          <option value="monte_carlo">Monte Carlo (Simulación)</option>
                        </Select>
                        
                        {/* Parámetros específicos del modelo */}
                        {userInputs.selectedModel === 'binomial' && (
                          <Input
                            label="Número de Pasos"
                            type="number"
                            min="10"
                            max="10000"
                            value={userInputs.nSteps}
                            onChange={(e) => updateUserInput('nSteps', e.target.value)}
                            placeholder="100"
                          />
                        )}
                        
                        {userInputs.selectedModel === 'monte_carlo' && (
                          <div className="space-y-2">
                            <Input
                              label="Número de Simulaciones"
                              type="number"
                              min="1000"
                              max="100000"
                              value={userInputs.nSimulations}
                              onChange={(e) => updateUserInput('nSimulations', e.target.value)}
                              placeholder="10000"
                            />
                            <Input
                              label="Semilla (opcional)"
                              type="number"
                              value={userInputs.seed || ''}
                              onChange={(e) => updateUserInput('seed', e.target.value || null)}
                              placeholder="Dejar vacío para aleatorio"
                            />
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
                      disabled={loading || !userInputs.riskFreeRate}
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
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {currentStep === 4 && (
          <motion.div
            key="step-4"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>Paso 5: Resultados del Análisis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Resultados del pricing */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-4 bg-zinc-800/50 rounded-xl">
                      <div className="text-2xl font-bold text-blue-400">
                        ${analysisResults.calculatedPrice?.toFixed(4) || '0.0000'}
                      </div>
                      <div className="text-sm text-zinc-400">Precio Calculado</div>
                    </div>
                    
                    <div className="text-center p-4 bg-zinc-800/50 rounded-xl">
                      <div className="text-2xl font-bold text-green-400">
                        ${yahooData.selectedOption?.last_price?.toFixed(4) || '0.0000'}
                      </div>
                      <div className="text-sm text-zinc-400">Precio de Mercado</div>
                    </div>
                    
                    <div className="text-center p-4 bg-zinc-800/50 rounded-xl">
                      <div className="text-2xl font-bold text-purple-400">
                        {analysisResults.calculatedPrice && yahooData.selectedOption?.last_price ? 
                          `${(((analysisResults.calculatedPrice - yahooData.selectedOption.last_price) / yahooData.selectedOption.last_price) * 100).toFixed(2)}%` : 
                          '0.00%'
                        }
                      </div>
                      <div className="text-sm text-zinc-400">Diferencia</div>
                    </div>
                    
                    <div className="text-center p-4 bg-zinc-800/50 rounded-xl">
                      <div className="text-2xl font-bold text-yellow-400">
                        {userInputs.selectedModel === 'black_scholes' ? 'Black-Scholes' : 
                         userInputs.selectedModel === 'binomial' ? 'Binomial' : 'Monte Carlo'}
                      </div>
                      <div className="text-sm text-zinc-400">Modelo Usado</div>
                    </div>
                  </div>

                  {/* Análisis de Griegas */}
                  {analysisResults.greeks && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Análisis de Griegas</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                          {[
                            { name: "Delta", value: analysisResults.greeks.delta, color: "blue", icon: <TrendingUp className="w-4 h-4" /> },
                            { name: "Gamma", value: analysisResults.greeks.gamma, color: "green", icon: <Activity className="w-4 h-4" /> },
                            { name: "Vega", value: analysisResults.greeks.vega, color: "purple", icon: <TrendingUp className="w-4 h-4" /> },
                            { name: "Theta", value: analysisResults.greeks.theta, color: "red", icon: <Clock className="w-4 h-4" /> },
                            { name: "Rho", value: analysisResults.greeks.rho, color: "yellow", icon: <TrendingDown className="w-4 h-4" /> },
                          ].map((greek, index) => (
                            <div key={index} className="text-center p-3 bg-zinc-800/30 rounded-lg">
                              <div className="flex items-center justify-center gap-1 mb-1 text-zinc-400">
                                {greek.icon}
                                <span className="text-xs">{greek.name}</span>
                              </div>
                              <div className={`text-lg font-semibold text-${greek.color}-400`}>
                                {parseFloat(greek.value).toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

                  {/* Botones de navegación */}
                  <div className="flex items-center justify-center gap-4">
                    <Button variant="ghost" onClick={prevStep}>
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