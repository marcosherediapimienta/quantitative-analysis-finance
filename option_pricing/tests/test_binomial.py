import unittest
from option_pricing.models.option import Option
from option_pricing.services.binomial_service import BinomialService

class TestBinomial(unittest.TestCase):
    def test_price(self):
        n_steps = 1000  # Hiperparámetro binomial
        # Valores esperados calculados previamente (puedes ajustarlos si cambias parámetros)
        expected_put = 5.57
        expected_call = 10.45
        tolerance = 0.01  # Margen de error aceptable

        # Test para opción put
        put_option = Option('put', 'european', 100, 100, 1, 0.2, 0.05)
        put_price = BinomialService.price(put_option, N=n_steps)
        print(f"Precio opción Put (Binomial, N={n_steps}): {put_price}")
        self.assertAlmostEqual(put_price, expected_put, delta=tolerance)

        # Test para opción call
        call_option = Option('call', 'european', 100, 100, 1, 0.2, 0.05)
        call_price = BinomialService.price(call_option, N=n_steps)
        print(f"Precio opción Call (Binomial, N={n_steps}): {call_price}")
        self.assertAlmostEqual(call_price, expected_call, delta=tolerance)

if __name__ == '__main__':
    unittest.main()
