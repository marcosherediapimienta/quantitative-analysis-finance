from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    OptionViewSet, PortfolioViewSet, PricingResultViewSet, 
    HealthCheckView, YahooFinanceTestView, YahooFinanceTickerView
)

app_name = 'option_pricing'

# Router para ViewSets
router = DefaultRouter()
router.register(r'options', OptionViewSet)
router.register(r'portfolios', PortfolioViewSet)
router.register(r'pricing-results', PricingResultViewSet)

urlpatterns = [
    # Health check endpoint
    path('health/', HealthCheckView.as_view(), name='health-check'),
    
    # Yahoo Finance endpoints
    path('yahoo-finance/test/', YahooFinanceTestView.as_view(), name='yahoo-finance-test'),
    path('yahoo-finance/ticker/', YahooFinanceTickerView.as_view(), name='yahoo-finance-ticker'),
    path('yahoo-finance/ticker/<str:symbol>/', YahooFinanceTickerView.as_view(), name='yahoo-finance-ticker-detail'),
    
    # Router URLs
    path('', include(router.urls)),
]

# URLs específicos adicionales si necesitas endpoints personalizados
urlpatterns += [
    # Ejemplos de endpoints adicionales:
    # path('custom-endpoint/', CustomView.as_view(), name='custom-endpoint'),
]
