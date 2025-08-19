from django.contrib import admin
from .models import (
    Stock, StockPrice, TechnicalIndicator, FinancialStatement, 
    FinancialMetric, RiskMetric, PortfolioOptimization, 
    PortfolioWeight, MarketData
)


@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'name', 'sector', 'market_cap', 'created_at']
    list_filter = ['sector', 'created_at']
    search_fields = ['symbol', 'name']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(StockPrice)
class StockPriceAdmin(admin.ModelAdmin):
    list_display = ['stock', 'date', 'close_price', 'volume']
    list_filter = ['date', 'stock']
    search_fields = ['stock__symbol', 'stock__name']
    date_hierarchy = 'date'


@admin.register(TechnicalIndicator)
class TechnicalIndicatorAdmin(admin.ModelAdmin):
    list_display = ['stock', 'indicator_type', 'period', 'value', 'date']
    list_filter = ['indicator_type', 'period', 'date']
    search_fields = ['stock__symbol']


@admin.register(FinancialStatement)
class FinancialStatementAdmin(admin.ModelAdmin):
    list_display = ['stock', 'statement_type', 'fiscal_year', 'fiscal_quarter', 'period_end']
    list_filter = ['statement_type', 'fiscal_year']
    search_fields = ['stock__symbol']


@admin.register(RiskMetric)
class RiskMetricAdmin(admin.ModelAdmin):
    list_display = ['stock', 'metric_type', 'value', 'calculation_date']
    list_filter = ['metric_type', 'calculation_date']
    search_fields = ['stock__symbol']


@admin.register(PortfolioOptimization)
class PortfolioOptimizationAdmin(admin.ModelAdmin):
    list_display = ['name', 'optimization_type', 'expected_return', 'volatility', 'sharpe_ratio']
    list_filter = ['optimization_type', 'optimization_date']
    search_fields = ['name']
