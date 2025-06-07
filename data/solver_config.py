class SolverConfig:
    """
        Configuration class for the PortfolioSolver class.
    """
    def __init__(
            self,
            max_weight_threshold: float = 0.3,
            risk_aversion: float = 0.3):
        self._max_weight_threshold = max_weight_threshold
        self._risk_aversion = risk_aversion # Risk aversion factor (0.0 = no risk, 1.0 = high risk)

    @property
    def max_weight_threshold(self) -> float:
        return self._max_weight_threshold

    @property
    def risk_aversion(self) -> float:
        return self._risk_aversion