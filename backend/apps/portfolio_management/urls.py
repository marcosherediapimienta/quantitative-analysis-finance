from django.urls import path, include
from rest_framework.routers import DefaultRouter

app_name = 'portfolio_management'

# Router para ViewSets (se expandirá cuando se creen las vistas)
router = DefaultRouter()

urlpatterns = [
    # Health check endpoint
    path('health/', lambda request: {'status': 'ok'}, name='health-check'),
    
    # Router URLs
    path('', include(router.urls)),
]

# URLs específicos adicionales
urlpatterns += [
    # TODO: Agregar endpoints específicos cuando se implementen las vistas
    # path('technical-analysis/', TechnicalAnalysisView.as_view(), name='technical-analysis'),
    # path('fundamental-analysis/', FundamentalAnalysisView.as_view(), name='fundamental-analysis'),
    # path('portfolio-optimization/', PortfolioOptimizationView.as_view(), name='portfolio-optimization'),
]
