from __future__ import annotations

from datetime import date, datetime
from typing import Any


def year_fraction(
    expiration_date: str,
    valuation_date: str | None = None,
    day_count: float = 365.0,
) -> float:
    expiration = datetime.strptime(expiration_date, "%Y-%m-%d").date()
    valuation = (
        datetime.strptime(valuation_date, "%Y-%m-%d").date()
        if valuation_date
        else date.today()
    )
    maturity = (expiration - valuation).days / day_count
    return max(maturity, 1e-8)


def prepare_contract(contract: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(contract)
    if "time_to_maturity" not in normalized and "expiration_date" in normalized:
        normalized["time_to_maturity"] = year_fraction(
            expiration_date=normalized["expiration_date"],
            valuation_date=normalized.get("valuation_date"),
        )
    normalized.pop("expiration_date", None)
    normalized.pop("valuation_date", None)
    return normalized
