import unittest
from option_pricing.models.option import Option
from option_pricing.services.monte_carlo_service import MonteCarloService

class TestMonteCarlo(unittest.TestCase):
    def test_price(self):
        n_sim = 10000  # Hiperparámetro Monte Carlo
        seed = 42      # Semilla para reproducibilidad
        expected_put = 5.64
        expected_call = 10.34
        tolerance = 0.01  # Monte Carlo puede tener más variabilidad

        # Test para opción put
        put_option = Option('put', 'european', 100, 100, 1, 0.2, 0.05)
        put_price = MonteCarloService.price(put_option, n_sim=n_sim, seed=seed)
        print(f"Precio opción Put (Monte Carlo, n_sim={n_sim}, seed={seed}): {put_price}")
        self.assertAlmostEqual(put_price, expected_put, delta=tolerance)

        # Test para opción call
        call_option = Option('call', 'european', 100, 100, 1, 0.2, 0.05)
        call_price = MonteCarloService.price(call_option, n_sim=n_sim, seed=seed)
        print(f"Precio opción Call (Monte Carlo, n_sim={n_sim}, seed={seed}): {call_price}")
        self.assertAlmostEqual(call_price, expected_call, delta=tolerance)

if __name__ == '__main__':
    unittest.main()
