from dataclasses import dataclass
from enum import Enum

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

    @classmethod
    def from_string(cls, value: str) -> "OptionType":
        normalized = (value or "call").strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid option type: {value}") from exc

@dataclass(frozen=True)
class OptionContract:
    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    steps: int = 1000
    option_type: OptionType = OptionType.CALL

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be > 0")
        if self.strike <= 0:
            raise ValueError("strike must be > 0")
        if self.time_to_maturity <= 0:
            raise ValueError("time_to_maturity must be > 0")
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.volatility <= 0:
            raise ValueError("volatility must be > 0")
