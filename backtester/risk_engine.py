class RiskEngine:
    def __init__(
        self,
        initial_capital: float,
        max_position_pct: float = 0.25,
        max_drawdown_limit: float = 0.15,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if not (0.0 < max_position_pct <= 1.0):
            raise ValueError("max_position_pct must be in (0, 1].")
        if not (0.0 < max_drawdown_limit < 1.0):
            raise ValueError("max_drawdown_limit must be in (0, 1).")

        self.max_position_pct = float(max_position_pct)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.peak_equity = float(initial_capital)
        self.circuit_breaker_active = False

    def evaluate_equity(self, cash: float, units: float, price: float) -> tuple[float, float]:
        equity = float(cash) + float(units) * float(price)
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = 0.0
        if self.peak_equity > 0.0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
        return equity, drawdown

    def check_drawdown_breach(self, drawdown: float) -> bool:
        if not self.circuit_breaker_active and (drawdown + 1e-12) >= self.max_drawdown_limit:
            self.circuit_breaker_active = True
            return True
        return False

    def calculate_position_allocation(self, cash: float, current_equity: float) -> float:
        if self.circuit_breaker_active or cash <= 0.0:
            return 0.0
        return min(float(cash), float(current_equity) * self.max_position_pct)