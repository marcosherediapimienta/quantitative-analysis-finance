import unittest
from backend.option_pricing.models.option import Option
from backend.option_pricing.controllers.portfolio_controller import PortfolioController

class TestPortfolioController(unittest.TestCase):
    def setUp(self):
        self.options = [
            Option('call', 'european', 100, 100, 1, 0.2, 0.05),
            Option('put', 'european', 100, 100, 1, 0.2, 0.05)
        ]
        self.controller = PortfolioController()
        # Valores esperados calculados previamente
        self.expected_black_scholes = 16.02
        self.expected_binomial = 16.02
        self.expected_monte_carlo = 15.99 
        self.seed = 42
        self.n_sim = 10000
        self.tolerance = 0.01

    def test_portfolio_value_black_scholes(self):
        value = self.controller.get_portfolio_value(self.options, model='black_scholes')
        print(f"Valor portfolio (Black-Scholes): {value}")
        self.assertAlmostEqual(value, self.expected_black_scholes, delta=self.tolerance)

    def test_portfolio_value_binomial(self):
        value = self.controller.get_portfolio_value(self.options, model='binomial')
        print(f"Valor portfolio (Binomial): {value}")
        self.assertAlmostEqual(value, self.expected_binomial, delta=self.tolerance)

    def test_portfolio_value_monte_carlo(self):
        value = self.controller.get_portfolio_value(self.options, model='monte_carlo', n_sim=self.n_sim, seed=self.seed)
        print(f"Valor portfolio (Monte Carlo, n_sim={self.n_sim}, seed={self.seed}): {value}")
        self.assertAlmostEqual(value, self.expected_monte_carlo, delta=self.tolerance)

    def test_portfolio_greeks_black_scholes(self):
        greeks = self.controller.get_portfolio_greeks(self.options, model='black_scholes')
        print(f"Greeks portfolio (Black-Scholes): {greeks}")
        self.assertEqual(len(greeks), 2)

if __name__ == '__main__':
    unittest.main()
