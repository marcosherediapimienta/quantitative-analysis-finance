import unittest
from option_pricing.models.option import Option
from option_pricing.services.binomial_service import BinomialService
from option_pricing.utils.plot_utils import plot_binomial_price_histogram, plot_binomial_sensitivity

class TestBinomial(unittest.TestCase):
    def test_price(self):
        n_steps = 1000  # Hiperparámetro binomial
        expected_put = 5.57
        expected_call = 10.45
        tolerance = 0.01

        # Test para opción put
        put_option = Option('put', 'european', 100, 100, 1, 0.2, 0.05)
        put_price = BinomialService.price(put_option, N=n_steps)
        print(f"Precio opción Put (Binomial, N={n_steps}): {put_price}")
        self.assertAlmostEqual(put_price, expected_put, delta=tolerance)
        # Plot histograma y sensibilidad
        plot_binomial_price_histogram(put_option, N=n_steps, n_sim=500, save_path='option_pricing/visualizations/test_binomial_put_hist.png')
        plot_binomial_sensitivity(put_option, 'spot', N=n_steps, save_path='option_pricing/visualizations/test_binomial_put_sens_spot.png')

        # Test para opción call
        call_option = Option('call', 'european', 100, 100, 1, 0.2, 0.05)
        call_price = BinomialService.price(call_option, N=n_steps)
        print(f"Precio opción Call (Binomial, N={n_steps}): {call_price}")
        self.assertAlmostEqual(call_price, expected_call, delta=tolerance)
        # Plot histograma y sensibilidad
        plot_binomial_price_histogram(call_option, N=n_steps, n_sim=500, save_path='option_pricing/visualizations/test_binomial_call_hist.png')
        plot_binomial_sensitivity(call_option, 'spot', N=n_steps, save_path='option_pricing/visualizations/test_binomial_call_sens_spot.png')

if __name__ == '__main__':
    unittest.main()
