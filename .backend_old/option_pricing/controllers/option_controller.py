from backend.option_pricing.services.black_scholes_service import BlackScholesService
from backend.option_pricing.models.option import Option

class OptionController:
    def get_option_price(self, option_data):
        option = Option(**option_data)
        return BlackScholesService.price(option)
