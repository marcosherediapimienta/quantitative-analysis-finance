import React from 'react';

/**
 * Componente simple para visualizar análisis de sensibilidad
 * En el futuro se puede integrar con librerías como Chart.js o Recharts
 */
const SensitivityChart = ({ data, sensitivityType, basePrice }) => {
  if (!data || !data.data_points || data.data_points.length === 0) {
    return (
      <div className="h-64 bg-zinc-900/50 rounded-lg flex items-center justify-center">
        <div className="text-center text-zinc-500">
          <div className="w-12 h-12 mx-auto mb-2 opacity-50">📊</div>
          <p>No hay datos para mostrar</p>
        </div>
      </div>
    );
  }

  const points = data.data_points;
  const minPrice = Math.min(...points.map(p => p.option_price));
  const maxPrice = Math.max(...points.map(p => p.option_price));
  const priceRange = maxPrice - minPrice;
  
  // Normalizar valores para el gráfico
  const normalizedPoints = points.map(point => ({
    ...point,
    normalizedPrice: priceRange > 0 ? (point.option_price - minPrice) / priceRange : 0.5,
    change: basePrice ? point.option_price - basePrice : 0,
    changePercent: basePrice ? ((point.option_price - basePrice) / basePrice) * 100 : 0
  }));

  // Obtener etiquetas para el eje X
  const getXAxisLabel = (value) => {
    switch (sensitivityType) {
      case 'spot':
        return `$${value.toFixed(2)}`;
      case 'strike':
        return `$${value.toFixed(2)}`;
      case 'volatility':
        return `${(value * 100).toFixed(1)}%`;
      case 'rate':
        return `${(value * 100).toFixed(2)}%`;
      case 'time':
        return `${(value * 365).toFixed(0)}d`;
      default:
        return value.toFixed(4);
    }
  };

  return (
    <div className="space-y-4">
      {/* Gráfico de Barras Simples */}
      <div className="h-64 bg-zinc-900/50 rounded-lg p-4 relative">
        <div className="absolute inset-0 flex items-end justify-between px-4 pb-4">
          {normalizedPoints.map((point, index) => (
            <div key={index} className="flex flex-col items-center">
              {/* Barra */}
              <div 
                className={`w-3 rounded-t-sm transition-all duration-300 ${
                  point.change > 0 
                    ? 'bg-green-500 hover:bg-green-400' 
                    : point.change < 0 
                    ? 'bg-red-500 hover:bg-red-400' 
                    : 'bg-zinc-500 hover:bg-zinc-400'
                }`}
                style={{ 
                  height: `${Math.max(point.normalizedPrice * 200, 4)}px`,
                  minHeight: '4px'
                }}
                title={`${getXAxisLabel(point.parameter_value)}: $${point.option_price.toFixed(4)} (${point.changePercent > 0 ? '+' : ''}${point.changePercent.toFixed(2)}%)`}
              />
              
              {/* Etiqueta del eje X */}
              <div className="text-xs text-zinc-400 mt-2 transform -rotate-45 origin-top-left">
                {getXAxisLabel(point.parameter_value)}
              </div>
            </div>
          ))}
        </div>
        
        {/* Etiquetas del eje Y */}
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-zinc-400 py-4">
          <span>${maxPrice.toFixed(4)}</span>
          <span>${((maxPrice + minPrice) / 2).toFixed(4)}</span>
          <span>${minPrice.toFixed(4)}</span>
        </div>
        
        {/* Línea de precio base */}
        {basePrice && (
          <div 
            className="absolute left-0 right-0 border-t-2 border-blue-400 border-dashed opacity-60"
            style={{ 
              bottom: `${((basePrice - minPrice) / priceRange) * 200}px`
            }}
          />
        )}
      </div>
      
      {/* Leyenda */}
      <div className="flex items-center justify-center gap-6 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-sm"></div>
          <span className="text-zinc-400">Aumento de precio</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded-sm"></div>
          <span className="text-zinc-400">Disminución de precio</span>
        </div>
        {basePrice && (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 border-t-2 border-blue-400 border-dashed"></div>
            <span className="text-zinc-400">Precio base</span>
          </div>
        )}
      </div>
      
      {/* Estadísticas Rápidas */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-zinc-800/30 rounded-lg">
          <div className="text-lg font-bold text-green-400">
            ${Math.max(...points.map(p => p.option_price)).toFixed(4)}
          </div>
          <div className="text-xs text-zinc-400">Precio Máximo</div>
        </div>
        
        <div className="p-3 bg-zinc-800/30 rounded-lg">
          <div className="text-lg font-bold text-red-400">
            ${Math.min(...points.map(p => p.option_price)).toFixed(4)}
          </div>
          <div className="text-xs text-zinc-400">Precio Mínimo</div>
        </div>
        
        <div className="p-3 bg-zinc-800/30 rounded-lg">
          <div className="text-lg font-bold text-blue-400">
            ${(Math.max(...points.map(p => p.option_price)) - Math.min(...points.map(p => p.option_price))).toFixed(4)}
          </div>
          <div className="text-xs text-zinc-400">Rango Total</div>
        </div>
      </div>
    </div>
  );
};

export default SensitivityChart;
