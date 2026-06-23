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
    TREND_FILTER = "Tendencia 4h bajista — no se abre posición"
    OK = "OK"


@dataclass
class RiskDecision:
    allowed: bool
    reason: BlockReason
    position_size_usd: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0


class RiskManager:
    def __init__(self, risk_cfg: dict, initial_capital: float, peak_value: float = None):
        self.max_position_size = risk_cfg["max_position_size"]
        self.stop_loss_pct = risk_cfg["stop_loss"]
        self.take_profit_pct = risk_cfg["take_profit"]
        self.daily_loss_limit = risk_cfg["daily_loss_limit"]
        self.max_drawdown = risk_cfg["max_drawdown"]

        self.atr_sl_mult = float(risk_cfg.get("atr_sl_multiplier", 0.0))
        self.atr_tp_mult = float(risk_cfg.get("atr_tp_multiplier", 0.0))

        self.trailing_breakeven_pct = float(risk_cfg.get("trailing_breakeven_pct", 1.5))
        self.trailing_activate_pct = float(risk_cfg.get("trailing_activate_pct", 3.0))
        self.trailing_distance_pct = float(risk_cfg.get("trailing_distance_pct", 1.5))

        self.initial_capital = initial_capital
        self.peak_value = peak_value if peak_value is not None else initial_capital
        self._daily_loss_triggered = False
        self._drawdown_triggered = False

    def update_portfolio_state(
        self,
        current_value: float,
        daily_pnl: float,
        initial_daily_value: float,
    ) -> None:
        if current_value > self.peak_value:
            self.peak_value = current_value

        daily_loss_pct = daily_pnl / initial_daily_value if initial_daily_value > 0 else 0
        drawdown_pct = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0

        if daily_loss_pct <= -self.daily_loss_limit:
            if not self._daily_loss_triggered:
                logger.critical(
                    f"CIRCUIT BREAKER DIARIO: pérdida={daily_loss_pct:.1%} "
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
        self._daily_loss_triggered = False
        logger.info("Límite de pérdida diaria reseteado para el nuevo día.")

    def _compute_sl_tp(self, current_price: float, atr_value: float) -> tuple:
        """Calcula SL y TP. Usa ATR si disponible, sino fallback a porcentajes fijos."""
        if atr_value > 0 and self.atr_sl_mult > 0 and self.atr_tp_mult > 0:
            sl_dist = atr_value * self.atr_sl_mult
            tp_dist = atr_value * self.atr_tp_mult
            # Límites de seguridad
            sl_dist = max(current_price * 0.005, min(current_price * 0.05, sl_dist))
            tp_dist = max(current_price * 0.01, min(current_price * 0.15, tp_dist))
        else:
            sl_dist = current_price * self.stop_loss_pct
            tp_dist = current_price * self.take_profit_pct

        return round(current_price - sl_dist, 6), round(current_price + tp_dist, 6)

    def evaluate_trade(
        self,
        signal: int,
        probability: float,
        confidence_threshold: float,
        current_price: float,
        available_capital: float,
        has_open_position: bool,
        atr_value: float = 0.0,
        trend_ok: bool = True,
    ) -> RiskDecision:
        if self._drawdown_triggered:
            return RiskDecision(allowed=False, reason=BlockReason.MAX_DRAWDOWN)
        if self._daily_loss_triggered:
            return RiskDecision(allowed=False, reason=BlockReason.DAILY_LOSS_LIMIT)
        if signal != 1:
            return RiskDecision(allowed=False, reason=BlockReason.LOW_CONFIDENCE)
        if probability < confidence_threshold:
            return RiskDecision(allowed=False, reason=BlockReason.LOW_CONFIDENCE)
        if has_open_position:
            return RiskDecision(allowed=False, reason=BlockReason.POSITION_OPEN)
        if not trend_ok:
            return RiskDecision(allowed=False, reason=BlockReason.TREND_FILTER)

        position_usd = available_capital * self.max_position_size
        if position_usd < 10:
            return RiskDecision(allowed=False, reason=BlockReason.INSUFFICIENT_CAPITAL)

        sl_price, tp_price = self._compute_sl_tp(current_price, atr_value)

        logger.info(
            f"Operación APROBADA | size=${position_usd:.2f} | "
            f"entrada={current_price:.4f} | SL={sl_price:.4f} | TP={tp_price:.4f} "
            f"({'ATR' if atr_value > 0 else 'fijo'})"
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
        if current_price <= stop_loss_price:
            loss_pct = (current_price / entry_price - 1) * 100
            logger.warning(f"STOP LOSS activado: precio={current_price:.4f} | pérdida={loss_pct:.2f}%")
            return "stop_loss"
        if current_price >= take_profit_price:
            gain_pct = (current_price / entry_price - 1) * 100
            logger.info(f"TAKE PROFIT alcanzado: precio={current_price:.4f} | ganancia={gain_pct:.2f}%")
            return "take_profit"
        return None

    def apply_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        current_sl: float,
        peak_price: float,
        trailing_active: bool,
    ) -> tuple:
        """
        Aplica lógica de trailing stop.
        Retorna: (exit_reason_or_None, new_sl, new_peak, trailing_now_active)
        """
        new_peak = max(peak_price if peak_price else current_price, current_price)
        pnl_pct = (current_price / entry_price - 1) * 100

        # Etapa 2: trailing dinámico activo
        if trailing_active or pnl_pct >= self.trailing_activate_pct:
            trailing_sl = round(new_peak * (1 - self.trailing_distance_pct / 100), 6)
            new_sl = max(current_sl, trailing_sl)
            if current_price <= new_sl:
                logger.info(f"TRAILING STOP activado: {current_price:.4f} <= {new_sl:.4f}")
                return "trailing_stop", new_sl, new_peak, True
            if new_sl > current_sl:
                logger.info(f"Trailing SL actualizado: {current_sl:.4f} → {new_sl:.4f}")
            return None, new_sl, new_peak, True

        # Etapa 1: mover a breakeven
        if pnl_pct >= self.trailing_breakeven_pct:
            new_sl = max(current_sl, entry_price)
            if current_price <= new_sl:
                return "trailing_stop", new_sl, new_peak, False
            if new_sl > current_sl:
                logger.info(f"SL movido a breakeven: {entry_price:.4f}")
            return None, new_sl, new_peak, False

        # Sin trailing aún
        return None, current_sl, new_peak, False
