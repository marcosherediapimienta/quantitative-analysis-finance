import unittest
from backend.option_pricing.models.option import Option
from backend.option_pricing.services.monte_carlo_service import MonteCarloService
from backend.option_pricing.utils.plot_utils import plot_mc_price_histogram, plot_mc_sensitivity

class TestMonteCarlo(unittest.TestCase):
    def test_price(self):
        n_sim = 10000  # Hiperparámetro Monte Carlo
        seed = 42      # Semilla para reproducibilidad
        expected_put = 5.64
        expected_call = 10.34
        tolerance = 0.01 

        # Test para opción put
        put_option = Option('put', 'european', 100, 100, 1, 0.2, 0.05)
        put_price = MonteCarloService.price(put_option, n_sim=n_sim, seed=seed)
        print(f"Precio opción Put (Monte Carlo, n_sim={n_sim}, seed={seed}): {put_price}")
        self.assertAlmostEqual(put_price, expected_put, delta=tolerance)
        # Plot histograma y sensibilidad
        plot_mc_price_histogram(put_option, n_sim=500, seed=seed, save_path='option_pricing/visualizations/test_mc_put_hist.png')
        plot_mc_sensitivity(put_option, 'spot', n_sim=500, seed=seed, save_path='option_pricing/visualizations/test_mc_put_sens_spot.png')

        # Test para opción call
        call_option = Option('call', 'european', 100, 100, 1, 0.2, 0.05)
        call_price = MonteCarloService.price(call_option, n_sim=n_sim, seed=seed)
        print(f"Precio opción Call (Monte Carlo, n_sim={n_sim}, seed={seed}): {call_price}")
        self.assertAlmostEqual(call_price, expected_call, delta=tolerance)
        # Plot histograma y sensibilidad
        plot_mc_price_histogram(call_option, n_sim=500, seed=seed, save_path='option_pricing/visualizations/test_mc_call_hist.png')
        plot_mc_sensitivity(call_option, 'spot', n_sim=500, seed=seed, save_path='option_pricing/visualizations/test_mc_call_sens_spot.png')

if __name__ == '__main__':
    unittest.main()
