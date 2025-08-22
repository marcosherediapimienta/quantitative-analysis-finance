import React, { useState } from 'react';

export const SensitivityChart = ({ data, analysisType, sensitivityType, basePrice }) => {
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // Usar analysisType o sensitivityType dependiendo de cuál esté disponible
  const type = analysisType || sensitivityType;

  // Validación de datos más robusta
  if (!data) {
    return (
      <div className="bg-zinc-800/50 rounded-lg p-8 text-center">
        <p className="text-zinc-400">No hay datos para mostrar</p>
      </div>
    );
  }

  // Si data no es un array, intentar extraer el array de datos
  let dataArray = data;
  if (!Array.isArray(data)) {
    // Si es un objeto, buscar propiedades que puedan contener el array
    if (data.data_points && Array.isArray(data.data_points)) {
      dataArray = data.data_points;
    } else if (data.data && Array.isArray(data.data)) {
      dataArray = data.data;
    } else if (data.results && Array.isArray(data.results)) {
      dataArray = data.results;
    } else if (data.scenarios && Array.isArray(data.scenarios)) {
      dataArray = data.scenarios;
    } else {
      console.error('Estructura de datos no reconocida:', data);
      return (
        <div className="bg-zinc-800/50 rounded-lg p-8 text-center">
          <p className="text-zinc-400">Error: Estructura de datos no válida</p>
          <p className="text-xs text-zinc-500 mt-2">
            Datos recibidos: {typeof data === 'object' ? JSON.stringify(Object.keys(data)) : typeof data}
          </p>
        </div>
      );
    }
  }

  if (!dataArray || dataArray.length === 0) {
    return (
      <div className="bg-zinc-800/50 rounded-lg p-8 text-center">
        <p className="text-zinc-400">No hay datos para mostrar</p>
      </div>
    );
  }

  // Preparar datos normalizados
  const prices = dataArray.map(d => d.option_price);
  const maxPrice = Math.max(...prices);
  const minPrice = Math.min(...prices);
  const priceRange = maxPrice - minPrice;

  const normalizedPoints = dataArray.map((point, index) => {
    const x = (index / (dataArray.length - 1)) * 780 + 10; // 10px padding
    const y = 280 - ((point.option_price - minPrice) / priceRange) * 260; // 20px padding arriba/abajo
    const change = point.option_price - basePrice;
    const changePercent = ((point.option_price - basePrice) / basePrice) * 100;
    
    return {
      x,
      y,
      price: point.option_price,
      parameter: point.parameter_value,
      change,
      changePercent,
      isFavorable: point.option_price > basePrice
    };
  });

  // Funciones helper
  const getParameterName = () => {
    switch (type) {
      case 'spot_price':
      case 'spot': 
        return 'Precio del Subyacente';
      case 'volatility': 
        return 'Volatilidad';
      case 'risk_free_rate':
      case 'rate': 
        return 'Tasa Libre de Riesgo';
      case 'time_to_maturity':
      case 'time': 
        return 'Tiempo hasta Vencimiento';
      default: 
        return 'Parámetro';
    }
  };

  const getParameterUnit = () => {
    switch (type) {
      case 'spot_price':
      case 'spot': 
        return '$';
      case 'volatility': 
        return '%';
      case 'risk_free_rate':
      case 'rate': 
        return '%';
      case 'time_to_maturity':
      case 'time': 
        return 'días';
      default: 
        return '';
    }
  };

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setMousePosition({ x: e.clientX, y: e.clientY });
    
    // Encontrar el punto más cercano
    let closestPoint = null;
    let minDistance = Infinity;
    
    normalizedPoints.forEach((point, index) => {
      const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2));
      if (distance < 30 && distance < minDistance) { // Radio de 30px para detectar
        minDistance = distance;
        closestPoint = index;
      }
    });
    
    setHoveredPoint(closestPoint);
  };

  // Generar etiquetas del eje Y
  const yLabels = [];
  for (let i = 0; i <= 5; i++) {
    const value = minPrice + (priceRange * (5 - i) / 5);
    yLabels.push(value);
  }

  // Generar etiquetas del eje X
  const xLabels = dataArray.filter((_, index) => index % Math.ceil(dataArray.length / 6) === 0);

  return (
    <div className="space-y-6">
      {/* Gráfico Principal */}
      <div className="bg-gradient-to-br from-zinc-900/95 to-zinc-800/95 rounded-xl p-6 border border-zinc-700/50">
        
        {/* Contenedor con título Y y gráfico */}
        <div className="flex items-center gap-6">
          
          {/* Título del Eje Y (rotado) */}
          <div className="flex items-center justify-center w-6">
            <div 
              className="text-sm font-semibold text-green-300 whitespace-nowrap transform -rotate-90"
              style={{ transformOrigin: 'center center' }}
            >
              Precio de la Opción ($)
            </div>
          </div>

          {/* Contenedor del gráfico */}
          <div className="flex-1">
            
            {/* Etiquetas del Eje Y */}
            <div className="flex">
              <div className="w-16 flex flex-col justify-between h-80 py-4">
                {yLabels.map((value, index) => (
                  <div key={index} className="text-xs text-zinc-400 text-right pr-2">
                    ${value.toFixed(2)}
                  </div>
                ))}
              </div>

              {/* SVG del Gráfico */}
              <div 
                className="flex-1 h-80 relative cursor-crosshair"
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoveredPoint(null)}
              >
                <svg 
                  width="100%" 
                  height="100%" 
                  viewBox="0 0 800 320"
                  className="overflow-visible"
                >
                  <defs>
                    {/* Gradiente para el área */}
                    <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#10b981" stopOpacity="0.3"/>
                      <stop offset="100%" stopColor="#10b981" stopOpacity="0.05"/>
                    </linearGradient>
                    
                    {/* Efecto glow */}
                    <filter id="glow">
                      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                      <feMerge> 
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                      </feMerge>
                    </filter>
                  </defs>

                  {/* Grid de fondo */}
                  {/* Líneas horizontales */}
                  {[0, 1, 2, 3, 4, 5].map(i => (
                    <line
                      key={`h-${i}`}
                      x1="0"
                      y1={20 + (i * 52)}
                      x2="800"
                      y2={20 + (i * 52)}
                      stroke="#374151"
                      strokeWidth="0.5"
                      opacity="0.3"
                    />
                  ))}
                  
                  {/* Líneas verticales */}
                  {[0, 1, 2, 3, 4, 5, 6].map(i => (
                    <line
                      key={`v-${i}`}
                      x1={i * 130 + 30}
                      y1="20"
                      x2={i * 130 + 30}
                      y2="280"
                      stroke="#374151"
                      strokeWidth="0.5"
                      opacity="0.3"
                    />
                  ))}

                  {/* Línea del precio teórico actual */}
                  {basePrice && (
                    <>
                      <line
                        x1="0"
                        y1={280 - ((basePrice - minPrice) / priceRange) * 260 + 20}
                        x2="800"
                        y2={280 - ((basePrice - minPrice) / priceRange) * 260 + 20}
                        stroke="#3b82f6"
                        strokeWidth="2"
                        strokeDasharray="8,4"
                        opacity="0.8"
                      />
                      <text
                        x="10"
                        y={280 - ((basePrice - minPrice) / priceRange) * 260 + 15}
                        fill="#3b82f6"
                        fontSize="11"
                        className="font-medium"
                      >
                        Precio Teórico: ${basePrice.toFixed(4)}
                      </text>
                    </>
                  )}

                  {/* Área bajo la curva */}
                  <polygon
                    fill="url(#areaGradient)"
                    points={`0,300 ${normalizedPoints.map(p => `${p.x},${p.y + 20}`).join(' ')} 800,300`}
                  />

                  {/* Línea principal */}
                  <polyline
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    filter="url(#glow)"
                    points={normalizedPoints.map(p => `${p.x},${p.y + 20}`).join(' ')}
                  />

                  {/* Puntos de datos */}
                  {normalizedPoints.map((point, index) => (
                    <g key={index}>
                      {/* Círculo invisible para detección */}
                      <circle
                        cx={point.x}
                        cy={point.y + 20}
                        r="20"
                        fill="transparent"
                        className="cursor-pointer"
                      />
                      
                      {/* Punto visible */}
                      <circle
                        cx={point.x}
                        cy={point.y + 20}
                        r={hoveredPoint === index ? "6" : "4"}
                        fill={point.isFavorable ? "#10b981" : "#ef4444"}
                        stroke="white"
                        strokeWidth="2"
                        className="transition-all duration-200"
                        style={{
                          filter: hoveredPoint === index ? 'drop-shadow(0 0 8px rgba(16, 185, 129, 0.6))' : 'none'
                        }}
                      />

                      {/* Líneas de hover */}
                      {hoveredPoint === index && (
                        <>
                          <line
                            x1={point.x}
                            y1="20"
                            x2={point.x}
                            y2="300"
                            stroke="#94a3b8"
                            strokeWidth="1"
                            strokeDasharray="4,4"
                            opacity="0.6"
                          />
                          <line
                            x1="0"
                            y1={point.y + 20}
                            x2="800"
                            y2={point.y + 20}
                            stroke="#94a3b8"
                            strokeWidth="1"
                            strokeDasharray="4,4"
                            opacity="0.6"
                          />
                        </>
                      )}
                    </g>
                  ))}
                </svg>

                {/* Tooltip */}
                {hoveredPoint !== null && (
                  <div 
                    className="fixed z-50 bg-zinc-900/95 backdrop-blur-sm border border-zinc-600/50 rounded-lg p-3 shadow-xl pointer-events-none min-w-[200px]"
                    style={{
                      left: mousePosition.x + 15,
                      top: mousePosition.y - 80,
                      transform: mousePosition.x > window.innerWidth - 250 ? 'translateX(-100%)' : 'none'
                    }}
                  >
                    <div className="text-xs text-zinc-400 mb-2">Escenario #{hoveredPoint + 1}</div>
                    
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between gap-4">
                        <span className="text-zinc-400">{getParameterName()}:</span>
                        <span className="font-mono text-blue-300">
                          {normalizedPoints[hoveredPoint].parameter}
                          {getParameterUnit()}
                        </span>
                      </div>
                      
                      <div className="flex justify-between gap-4">
                        <span className="text-zinc-400">Precio:</span>
                        <span className="font-mono text-white">
                          ${normalizedPoints[hoveredPoint].price.toFixed(4)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between gap-4">
                        <span className="text-zinc-400">Cambio:</span>
                        <span className={`font-mono ${
                          normalizedPoints[hoveredPoint].change > 0 ? 'text-green-400' : 
                          normalizedPoints[hoveredPoint].change < 0 ? 'text-red-400' : 'text-zinc-400'
                        }`}>
                          {normalizedPoints[hoveredPoint].change > 0 ? '+' : ''}
                          ${normalizedPoints[hoveredPoint].change.toFixed(4)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between gap-4">
                        <span className="text-zinc-400">Cambio %:</span>
                        <span className={`font-mono ${
                          normalizedPoints[hoveredPoint].changePercent > 0 ? 'text-green-400' : 
                          normalizedPoints[hoveredPoint].changePercent < 0 ? 'text-red-400' : 'text-zinc-400'
                        }`}>
                          {normalizedPoints[hoveredPoint].changePercent > 0 ? '+' : ''}
                          {normalizedPoints[hoveredPoint].changePercent.toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Etiquetas del Eje X */}
            <div className="flex justify-between px-16 mt-2">
              {xLabels.map((point, index) => (
                <div key={index} className="text-xs text-zinc-400 text-center">
                  {getParameterUnit() === '%' ? `${point.parameter_value}%` : 
                   getParameterUnit() === 'días' ? `${point.parameter_value}d` :
                   `$${point.parameter_value.toFixed(2)}`}
                </div>
              ))}
            </div>

            {/* Título del Eje X */}
            <div className="text-center mt-4">
              <div className="text-sm font-semibold text-blue-300">
                {getParameterName()}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Estadísticas */}
      <div className="grid grid-cols-3 gap-4">
        {/* Mejor Escenario */}
        <div className="bg-gradient-to-br from-green-900/20 to-green-800/20 rounded-lg p-4 border border-green-700/30">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <h4 className="text-sm font-semibold text-green-300">Mejor Escenario</h4>
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            ${Math.max(...prices).toFixed(4)}
          </div>
          <div className="text-xs text-green-400">
            +${(Math.max(...prices) - basePrice).toFixed(4)} vs teórico
          </div>
        </div>

        {/* Peor Escenario */}
        <div className="bg-gradient-to-br from-red-900/20 to-red-800/20 rounded-lg p-4 border border-red-700/30">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <h4 className="text-sm font-semibold text-red-300">Peor Escenario</h4>
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            ${Math.min(...prices).toFixed(4)}
          </div>
          <div className="text-xs text-red-400">
            ${(Math.min(...prices) - basePrice).toFixed(4)} vs teórico
          </div>
        </div>

        {/* Rango de Variación */}
        <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 rounded-lg p-4 border border-blue-700/30">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <h4 className="text-sm font-semibold text-blue-300">Rango Total</h4>
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            ${(maxPrice - minPrice).toFixed(4)}
          </div>
          <div className="text-xs text-blue-400">
            Amplitud del análisis
          </div>
        </div>
      </div>

      {/* Leyenda */}
      <div className="flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded-full"></div>
          <span className="text-zinc-300">Escenarios favorables</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded-full"></div>
          <span className="text-zinc-300">Escenarios desfavorables</span>
        </div>
        {basePrice && (
          <div className="flex items-center gap-2">
            <div className="w-6 h-1 bg-blue-500 opacity-80" style={{ clipPath: 'polygon(0 0, 8px 0, 12px 100%, 4px 100%)' }}></div>
            <span className="text-zinc-300">Precio teórico: ${basePrice.toFixed(4)}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default SensitivityChart;