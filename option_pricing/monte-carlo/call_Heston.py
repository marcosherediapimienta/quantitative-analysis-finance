import numpy as np
import matplotlib.pyplot as plt

def heston_monte_carlo(S0, K, T, r, q, V0, kappa, theta, xi, rho, n_simulations, n_steps):
    np.random.seed(42)
    dt = T / n_steps
    
    # Arrays para almacenar trayectorias
    S = np.zeros((n_steps + 1, n_simulations))
    V = np.zeros((n_steps + 1, n_simulations))
    S[0] = S0
    V[0] = V0
    
    # Generación de números aleatorios correlacionados
    for t in range(1, n_steps + 1):
        Z1 = np.random.normal(0, 1, n_simulations)
        Z2 = np.random.normal(0, 1, n_simulations)
        ZV = Z1
        ZS = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        # Evolución de la volatilidad (full truncation para V > 0)
        V[t] = np.maximum(V[t-1] + kappa * (theta - np.maximum(V[t-1], 0)) * dt + \
               xi * np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * ZV, 0)
        
        # Evolución del precio
        S[t] = S[t-1] * np.exp((r - q - 0.5 * np.maximum(V[t-1], 0)) * dt + \
               np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * ZS)
    
    return S, V

# Parámetros
S0, K, T, r, q = 123, 120, 38/365, 0.0421, 0.04
V0, kappa, theta, xi, rho = 0.04, 2.0, 0.0625, 0.3, -0.6
n_simulations, n_steps = 50_000, 252  # 252 pasos (días bursátiles)

S, V = heston_monte_carlo(S0, K, T, r, q, V0, kappa, theta, xi, rho, n_simulations, n_steps)

# Payoff de la call europea
payoffs = np.maximum(S[-1] - K, 0)
price_mc = np.exp(-r * T) * np.mean(payoffs)

print(f"Precio Monte Carlo (Heston): {price_mc:.2f}")

# Guardar los gráficos como archivos de imagen
plt.figure(figsize=(12, 6))

# Primer gráfico: Trayectorias (Heston)
plt.subplot(1, 2, 1)
plt.plot(S[:, :10])  # Primeras 10 trayectorias
plt.title("Trayectorias (Heston)")
plt.xlabel("Tiempo")
plt.ylabel("Precio")

# Segundo gráfico: Trayectorias de Volatilidad
plt.subplot(1, 2, 2)
plt.plot(V[:, :10])
plt.title("Trayectorias de Volatilidad")
plt.xlabel("Tiempo")
plt.ylabel("V")

# Ajustar y guardar la figura
plt.tight_layout()
plt.savefig('trayectorias_heston.png') 
print("Gráficos guardados como 'trayectorias_heston.png'.")
