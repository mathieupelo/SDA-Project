class SolverConfig:
    def __init__(
            self,
            penalty_factor: float = 0.00001,
            max_weight_threshold: float = 0.3,
            risk_aversion: float = 0.3):
        self._penalty_factor = penalty_factor
        self._max_weight_threshold = max_weight_threshold
        self._risk_aversion = risk_aversion

    @property
    def penalty_factor(self) -> float:
        return self._penalty_factor

    @property
    def max_weight_threshold(self) -> float:
        return self._max_weight_threshold

    @property
    def risk_aversion(self) -> float:
        return self._risk_aversion