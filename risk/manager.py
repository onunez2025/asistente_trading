import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class BlockReason(Enum):
    DAILY_LOSS_LIMIT = "Límite de pérdida diaria alcanzado"
    MAX_DRAWDOWN = "Circuit breaker: drawdown máximo alcanzado"
    POSITION_OPEN = "Ya hay una posición abierta"
    LOW_CONFIDENCE = "Confianza del modelo insuficiente"
    INSUFFICIENT_CAPITAL = "Capital insuficiente para operar"
    OK = "OK"


@dataclass
class RiskDecision:
    allowed: bool
    reason: BlockReason
    position_size_usd: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0


class RiskManager:
    """
    Controla cuándo y cuánto se puede operar.
    Es la última barrera antes de enviar una orden al exchange.
    """

    def __init__(self, risk_cfg: dict, initial_capital: float):
        self.max_position_size = risk_cfg["max_position_size"]   # 0.15 = 15%
        self.stop_loss_pct = risk_cfg["stop_loss"]               # 0.02 = 2%
        self.take_profit_pct = risk_cfg["take_profit"]           # 0.04 = 4%
        self.daily_loss_limit = risk_cfg["daily_loss_limit"]     # 0.05 = 5%
        self.max_drawdown = risk_cfg["max_drawdown"]             # 0.20 = 20%

        self.initial_capital = initial_capital
        self.peak_value = initial_capital
        self._daily_loss_triggered = False
        self._drawdown_triggered = False

    def update_portfolio_state(
        self,
        current_value: float,
        daily_pnl: float,
        initial_daily_value: float,
    ) -> None:
        """Actualiza el estado interno con los valores actuales del portfolio."""
        if current_value > self.peak_value:
            self.peak_value = current_value

        daily_loss_pct = daily_pnl / initial_daily_value if initial_daily_value > 0 else 0
        drawdown_pct = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0

        if daily_loss_pct <= -self.daily_loss_limit:
            if not self._daily_loss_triggered:
                logger.critical(
                    f"CIRCUIT BREAKER DIARIO: pérdida del día={daily_loss_pct:.1%} "
                    f"supera límite={self.daily_loss_limit:.1%}. Bot detenido hasta mañana."
                )
            self._daily_loss_triggered = True

        if drawdown_pct >= self.max_drawdown:
            if not self._drawdown_triggered:
                logger.critical(
                    f"CIRCUIT BREAKER TOTAL: drawdown={drawdown_pct:.1%} "
                    f"supera límite={self.max_drawdown:.1%}. Bot detenido."
                )
            self._drawdown_triggered = True

    def reset_daily_limit(self) -> None:
        """Llama esto al inicio de cada día para resetear el límite diario."""
        self._daily_loss_triggered = False
        logger.info("Límite de pérdida diaria reseteado para el nuevo día.")

    def evaluate_trade(
        self,
        signal: int,
        probability: float,
        confidence_threshold: float,
        current_price: float,
        available_capital: float,
        has_open_position: bool,
    ) -> RiskDecision:
        """
        Decide si se puede abrir una operación nueva.
        Retorna RiskDecision con todos los parámetros de la orden.
        """
        # 1. Circuit breakers
        if self._drawdown_triggered:
            return RiskDecision(allowed=False, reason=BlockReason.MAX_DRAWDOWN)

        if self._daily_loss_triggered:
            return RiskDecision(allowed=False, reason=BlockReason.DAILY_LOSS_LIMIT)

        # 2. Solo operar si la señal es de compra
        if signal != 1:
            return RiskDecision(allowed=False, reason=BlockReason.LOW_CONFIDENCE)

        # 3. Confianza mínima del modelo
        if probability < confidence_threshold:
            return RiskDecision(allowed=False, reason=BlockReason.LOW_CONFIDENCE)

        # 4. Sin posición abierta
        if has_open_position:
            return RiskDecision(allowed=False, reason=BlockReason.POSITION_OPEN)

        # 5. Capital suficiente para al menos una operación mínima
        position_usd = available_capital * self.max_position_size
        if position_usd < 10:
            return RiskDecision(allowed=False, reason=BlockReason.INSUFFICIENT_CAPITAL)

        # 6. Calcular Stop Loss y Take Profit
        sl_price = round(current_price * (1 - self.stop_loss_pct), 6)
        tp_price = round(current_price * (1 + self.take_profit_pct), 6)

        logger.info(
            f"Operación APROBADA | tamaño=${position_usd:.2f} | "
            f"entrada={current_price:.4f} | SL={sl_price:.4f} | TP={tp_price:.4f}"
        )

        return RiskDecision(
            allowed=True,
            reason=BlockReason.OK,
            position_size_usd=round(position_usd, 2),
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
        )

    def check_exit_conditions(
        self,
        current_price: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> Optional[str]:
        """
        Verifica si se debe cerrar una posición abierta.
        Retorna el motivo de cierre o None si debe seguir abierta.
        """
        if current_price <= stop_loss_price:
            loss_pct = (current_price / entry_price - 1) * 100
            logger.warning(f"STOP LOSS activado: precio={current_price:.4f} | pérdida={loss_pct:.2f}%")
            return "stop_loss"

        if current_price >= take_profit_price:
            gain_pct = (current_price / entry_price - 1) * 100
            logger.info(f"TAKE PROFIT alcanzado: precio={current_price:.4f} | ganancia={gain_pct:.2f}%")
            return "take_profit"

        return None
