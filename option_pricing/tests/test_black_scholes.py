import unittest
from option_pricing.models.option import Option
from option_pricing.services.black_scholes_service import BlackScholesService

class TestBlackScholes(unittest.TestCase):
    def test_price(self):
        # Valores esperados calculados previamente
        expected_put = 5.57
        expected_call = 10.45
        tolerance = 0.01

        # Test para opción put
        put_option = Option('put', 'european', 100, 100, 1, 0.2, 0.05)
        put_price = BlackScholesService.price(put_option)
        print(f"Precio opción Put (Black-Scholes): {put_price}")
        self.assertAlmostEqual(put_price, expected_put, delta=tolerance)

        # Test para opción call
        call_option = Option('call', 'european', 100, 100, 1, 0.2, 0.05)
        call_price = BlackScholesService.price(call_option)
        print(f"Precio opción Call (Black-Scholes): {call_price}")
        self.assertAlmostEqual(call_price, expected_call, delta=tolerance)

if __name__ == '__main__':
    unittest.main()
