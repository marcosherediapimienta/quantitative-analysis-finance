from django.contrib import admin
from django.utils.html import format_html
from .models import Option, Greeks, Portfolio, PortfolioPosition, PricingResult


@admin.register(Option)
class OptionAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'type', 'style', 'spot', 'strike', 'maturity', 
        'volatility', 'rate', 'is_in_the_money', 'created_at'
    ]
    list_filter = ['type', 'style', 'created_at']
    search_fields = ['name']
    readonly_fields = ['created_at', 'updated_at', 'is_in_the_money', 'moneyness']
    
    fieldsets = (
        ('Información General', {
            'fields': ('name', 'type', 'style')
        }),
        ('Parámetros de la Opción', {
            'fields': ('spot', 'strike', 'maturity', 'volatility', 'rate', 'market_price')
        }),
        ('Información Calculada', {
            'fields': ('is_in_the_money', 'moneyness'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('greeks')


@admin.register(Greeks)
class GreeksAdmin(admin.ModelAdmin):
    list_display = [
        'option_name', 'delta', 'gamma', 'vega', 'theta', 'rho', 
        'pricing_model', 'calculated_at'
    ]
    list_filter = ['pricing_model', 'calculated_at']
    search_fields = ['option__name']
    readonly_fields = ['calculated_at']
    
    def option_name(self, obj):
        return obj.option.name
    option_name.short_description = 'Option'
    option_name.admin_order_field = 'option__name'


class PortfolioPositionInline(admin.TabularInline):
    model = PortfolioPosition
    extra = 1
    fields = ['option', 'quantity', 'entry_price', 'is_long', 'notional_value']
    readonly_fields = ['is_long', 'notional_value']


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = ['name', 'total_positions', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'total_positions']
    inlines = [PortfolioPositionInline]
    
    fieldsets = (
        ('Información General', {
            'fields': ('name', 'description')
        }),
        ('Estadísticas', {
            'fields': ('total_positions',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(PortfolioPosition)
class PortfolioPositionAdmin(admin.ModelAdmin):
    list_display = [
        'portfolio_name', 'option_name', 'quantity', 'entry_price', 
        'position_type', 'notional_value', 'created_at'
    ]
    list_filter = ['created_at', 'portfolio']
    search_fields = ['portfolio__name', 'option__name']
    readonly_fields = ['created_at', 'updated_at', 'is_long', 'is_short', 'notional_value']
    
    def portfolio_name(self, obj):
        return obj.portfolio.name
    portfolio_name.short_description = 'Portfolio'
    portfolio_name.admin_order_field = 'portfolio__name'
    
    def option_name(self, obj):
        return obj.option.name
    option_name.short_description = 'Option'
    option_name.admin_order_field = 'option__name'
    
    def position_type(self, obj):
        if obj.is_long:
            return format_html('<span style="color: green;">Long</span>')
        elif obj.is_short:
            return format_html('<span style="color: red;">Short</span>')
        return 'Neutral'
    position_type.short_description = 'Type'


@admin.register(PricingResult)
class PricingResultAdmin(admin.ModelAdmin):
    list_display = [
        'option_name', 'model', 'calculated_price', 'implied_volatility',
        'n_steps', 'n_simulations', 'calculated_at'
    ]
    list_filter = ['model', 'calculated_at']
    search_fields = ['option__name']
    readonly_fields = ['calculated_at']
    
    fieldsets = (
        ('Opción', {
            'fields': ('option',)
        }),
        ('Resultado del Pricing', {
            'fields': ('model', 'calculated_price', 'implied_volatility')
        }),
        ('Parámetros del Modelo', {
            'fields': ('n_steps', 'n_simulations'),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('calculated_at',),
            'classes': ('collapse',)
        }),
    )
    
    def option_name(self, obj):
        return obj.option.name
    option_name.short_description = 'Option'
    option_name.admin_order_field = 'option__name'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('option')


# Personalización del admin site
admin.site.site_header = 'Quantitative Finance Administration'
admin.site.site_title = 'Quantitative Finance Admin'
admin.site.index_title = 'Welcome to Quantitative Finance Administration'
