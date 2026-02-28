from .metrics import value_at_risk_expected_shortfall

def simulate_monte_carlo_portfolio_pricing(*args, **kwargs):
    from .simulation import simulate_monte_carlo_portfolio_pricing as _simulate

    return _simulate(*args, **kwargs)

__all__ = ["value_at_risk_expected_shortfall", "simulate_monte_carlo_portfolio_pricing"]
