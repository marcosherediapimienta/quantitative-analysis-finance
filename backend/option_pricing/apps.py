from django.apps import AppConfig


class OptionPricingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'option_pricing'
    verbose_name = 'Option Pricing'
    
    def ready(self):
        # Importar señales aquí si las necesitas
        pass
