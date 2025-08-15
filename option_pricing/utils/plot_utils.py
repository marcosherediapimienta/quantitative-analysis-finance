import matplotlib.pyplot as plt
from option_pricing.models.option import Option

def plot_payoff(option: Option):
    import numpy as np
    S = np.linspace(option.strike * 0.5, option.strike * 1.5, 100)
    if option.type == 'call':
        payoff = np.maximum(S - option.strike, 0)
    elif option.type == 'put':
        payoff = np.maximum(option.strike - S, 0)
    else:
        raise ValueError('Tipo de opción no soportado')
    plt.figure(figsize=(8, 5))
    plt.plot(S, payoff, label=f'{option.type.capitalize()} Payoff')
    plt.xlabel('Precio del subyacente (S)')
    plt.ylabel('Payoff')
    plt.title(f'Payoff de opción {option.type} ({option.style})')
    plt.legend()
    plt.grid(True)
    plt.show()
