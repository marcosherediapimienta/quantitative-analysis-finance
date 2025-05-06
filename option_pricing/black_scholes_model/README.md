# Black-Scholes Options Analysis Framework

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Framework](#theoretical-framework)
3. [Option Pricing Model](#option-pricing-model)
4. [Risk Metrics (Greeks)](#risk-metrics-greeks)
5. [Volatility Analysis](#volatility-analysis)
6. [Price Affecting Factors](#price-affecting-factors)
7. [Trading Strategies](#trading-strategies)
8. [References](#references)

## Introduction

### What are Options?
Options are financial derivatives that give the holder the right, but not the obligation, to buy (call) or sell (put) an underlying asset at a predetermined price (strike price) before a specified date (expiration date).

### Key Option States
| State | Call Option | Put Option |
|-------|-------------|------------|
| ITM (In-the-money) | S > K | S < K |
| ATM (At-the-money) | S ≈ K | S ≈ K |
| OTM (Out-of-the-money) | S < K | S > K |

Where:
- S = Current asset price
- K = Strike price

## Theoretical Framework

### Black-Scholes Model Assumptions
1. **Market Efficiency**
   - No arbitrage opportunities
   - Perfect market conditions

2. **Asset Behavior**
   - Geometric Brownian motion
   - Continuous price movements
   - No jumps or gaps

3. **Market Parameters**
   - Constant risk-free rate
   - Constant volatility
   - No transaction costs
   - No dividends

## Option Pricing Model

### Core Formulas

#### Call Option Price
```
C = S * N(d1) - K * e^(-rT) * N(d2)
```

#### Put Option Price
```
P = K * e^(-rT) * N(-d2) - S * N(-d1)
```

#### Auxiliary Calculations
```
d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
d2 = d1 - σ√T
```

### Input Parameters
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Asset Price | S | Current price of underlying asset |
| Strike Price | K | Exercise price of the option |
| Time to Expiry | T | Time until expiration (in years) |
| Risk-free Rate | r | Annual risk-free interest rate |
| Volatility | σ | Annualized volatility of returns |
| Normal CDF | N() | Cumulative normal distribution function |

## Risk Metrics (Greeks)

### First-Order Greeks

#### Delta (Δ)
- **Definition**: Price sensitivity to underlying asset price
- **Call**: Δ = N(d1)
- **Put**: Δ = N(d1) - 1
- **Range**: [-1, 1]
- **Interpretation**: Probability of option expiring ITM

#### Theta (Θ)
- **Definition**: Price sensitivity to time decay
- **Call**: Θ = -S * σ * N'(d1) / (2√T) - r * K * e^(-rT) * N(d2)
- **Put**: Θ = -S * σ * N'(d1) / (2√T) + r * K * e^(-rT) * N(-d2)
- **Interpretation**: Daily price decay

#### Rho (ρ)
- **Definition**: Price sensitivity to interest rate
- **Call**: ρ = K * T * e^(-rT) * N(d2)
- **Put**: ρ = -K * T * e^(-rT) * N(-d2)
- **Interpretation**: Impact of rate changes

### Second-Order Greeks

#### Gamma (Γ)
- **Definition**: Delta sensitivity to price changes
- **Formula**: Γ = N'(d1) / (S * σ * √T)
- **Characteristics**: Always positive, maximum at ATM

#### Vega (ν)
- **Definition**: Price sensitivity to volatility
- **Formula**: ν = S * √T * N'(d1)
- **Characteristics**: Always positive, maximum at ATM

## Volatility Analysis

### Types of Volatility

1. **Historical Volatility**
   - Based on past price movements
   - Calculated using standard deviation of returns
   - Used for risk assessment

2. **Implied Volatility**
   - Derived from option market prices
   - Reflects market's future volatility expectations
   - Key for option pricing

### Volatility Patterns

#### Volatility Smile
- Higher implied volatility for OTM and ITM options
- Lower implied volatility for ATM options
- Reflects market's expectation of extreme moves

## Price Affecting Factors

### 1. Underlying Asset Price (S)
- **Call Options**: 
  - Direct positive relationship
  - Higher S → Higher call price
  - Delta measures this sensitivity
- **Put Options**:
  - Direct negative relationship
  - Higher S → Lower put price
  - Delta measures this sensitivity

### 2. Strike Price (K)
- **Call Options**:
  - Inverse relationship
  - Higher K → Lower call price
  - Affects moneyness of option
- **Put Options**:
  - Direct relationship
  - Higher K → Higher put price
  - Affects moneyness of option

### 3. Time to Expiration (T)
- **Both Options**:
  - Longer T → Higher option price
  - Non-linear relationship
  - Theta measures time decay
  - Decay accelerates near expiration
  - Expressed in years (e.g., 30/365 for 30 days)

### 4. Volatility (σ)
- **Both Options**:
  - Direct relationship
  - Higher σ → Higher option price
  - Vega measures this sensitivity
  - Most significant for ATM options
  - Affects probability of exercise

### 5. Risk-Free Rate (r)
- **Call Options**:
  - Direct relationship
  - Higher r → Higher call price
  - Rho measures this sensitivity
- **Put Options**:
  - Inverse relationship
  - Higher r → Lower put price
  - Rho measures this sensitivity

### 6. Dividends
- **Call Options**:
  - Inverse relationship
  - Higher dividends → Lower call price
  - Affects forward price
- **Put Options**:
  - Direct relationship
  - Higher dividends → Higher put price
  - Affects forward price

### 7. Market Conditions
- **Liquidity**:
  - Higher liquidity → Lower spreads
  - Better price discovery
- **Market Sentiment**:
  - Affects implied volatility
  - Influences option demand

## Trading Strategies

### Directional Strategies

#### Bullish Strategies
1. **Bull Call Spread**
   - Buy ITM call
   - Sell OTM call
   - Limited risk, limited reward

2. **Covered Call**
   - Buy underlying asset
   - Sell OTM call
   - Income generation

#### Bearish Strategies
1. **Bear Put Spread**
   - Buy ITM put
   - Sell OTM put
   - Limited risk, limited reward

2. **Protective Put**
   - Buy underlying asset
   - Buy OTM put
   - Downside protection

### Volatility Strategies

1. **Straddle**
   - Buy ATM call
   - Buy ATM put
   - Profits from large moves

2. **Strangle**
   - Buy OTM call
   - Buy OTM put
   - Cheaper than straddle

3. **Butterfly**
   - Buy ITM call
   - Sell 2 ATM calls
   - Buy OTM call
   - Limited risk, limited reward

## References

### Academic Papers
1. Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
2. Merton, R. C. (1973). Theory of Rational Option Pricing

### Books
1. Hull, J. C. (2018). Options, Futures, and Other Derivatives
2. Natenberg, S. (2015). Option Volatility and Pricing
3. Taleb, N. N. (1997). Dynamic Hedging: Managing Vanilla and Exotic Options
