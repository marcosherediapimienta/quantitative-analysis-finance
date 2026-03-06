from typing import Any, Dict, Optional
from ..tools import IVContract, ImpliedVolatilitySolver, SolverMethod

class IVAnalyzer:
    def __init__(
        self,
        solver: Optional[ImpliedVolatilitySolver] = None,
        method: SolverMethod = "newton",
        **solver_kwargs: Any,
    ) -> None:
        self.solver = solver or ImpliedVolatilitySolver(method=method, **solver_kwargs)

    def implied_volatility(self, contract: IVContract) -> float:
        result = self.solver.solve(contract)
        return float(result["implied_volatility"])

    def analyze(self, contract: IVContract) -> Dict[str, Any]:
        return self.solver.solve(contract)
