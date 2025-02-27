# 🚀 Portfolio Management Project

Este proyecto tiene como objetivo proporcionar herramientas y modelos para gestionar carteras de inversión de manera eficiente. Este proyecto utiliza la API de Yahoo Finance para descargar los datos históricos de precios de los activos. La API se accede a través de la biblioteca yfinance, que permite obtener los precios ajustados de cierre y realizar análisis de datos financieros.

## 🧑‍💻 Estructura del Proyecto

La estructura del repositorio se divide en varios módulos para cubrir diferentes aspectos de la gestión de carteras:

- **`Fundamental_Analysis.py`**: Realiza un análisis de los activos a partir de datos financieros fundamentales como ratios, estados financieros, etc.

- **`Technical_Analysis.py`**: Realiza un análisis técnico utilizando indicadores y patrones de precios históricos para prever movimientos futuros.

- **`Market_Risk_Analysis.py`**: Analiza el riesgo asociado a los activos en el mercado, utilizando métricas como la volatilidad y la correlación.

- **`CAPM (Capital Asset Pricing Model).py`**: Implementación del **CAPM**, utilizado para calcular el retorno esperado de un activo según su riesgo sistemático (beta).

- **`Portfolio_Optimization.py`**: Implementación de técnicas de optimización para construir una cartera de inversión que maximice el rendimiento o minimice el riesgo, dependiendo del objetivo.

## Requisitos 

- **Python 3.x**

Además, este proyecto depende de varias librerías en Python. A continuación, se detallan las librerías principales utilizadas:

- **NumPy**: Para la manipulación eficiente de arrays y matrices.
- **Pandas**: Para la manipulación de datos y análisis de series temporales.
- **yfinance**: Para descargar los datos históricos de precios de activos desde Yahoo Finance.
- **Matplotlib**: Para la creación de gráficos y visualizaciones.
- **Seaborn**: Para mejorar las visualizaciones estadísticas.
- **Scipy**: Para la optimización de la cartera y otros cálculos matemáticos.
- **Pandas_ta**: Para indicadores técnicos adicionales como RSI, MACD, etc.

## Instalación

1. Clona este repositorio en tu máquina local:

https://github.com/marcosherediapimienta/Portfolio-Management.git