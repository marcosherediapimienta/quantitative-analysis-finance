from typing import Any, Dict, Optional

from ..tools.config import DEFAULT_METHOD
from ..tools.types import IVContract, SolverMethod
from .components.solver import ImpliedVolatilitySolver


class IVAnalyzer:
    def __init__(
        self,
        solver: Optional[ImpliedVolatilitySolver] = None,
        method: SolverMethod = DEFAULT_METHOD,
        **solver_kwargs: Any,
    ) -> None:
        self.solver = solver or ImpliedVolatilitySolver(method=method, **solver_kwargs)

    def implied_volatility(self, contract: IVContract) -> float:
        result = self.solver.solve(contract)
        return float(result["implied_volatility"])

    def analyze(self, contract: IVContract) -> Dict[str, Any]:
        return self.solver.solve(contract)
