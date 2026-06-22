import logging
import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Envía mensajes al usuario vía Telegram Bot API."""

    def __init__(self, token: str, chat_id: str, enabled: bool = False):
        self.token = token
        self.chat_id = str(chat_id)
        self.enabled = enabled and bool(token) and bool(chat_id)

        if enabled and not self.enabled:
            logger.warning(
                "Telegram habilitado en settings pero falta token o chat_id. "
                "Las notificaciones están desactivadas."
            )

    def _send(self, text: str) -> bool:
        if not self.enabled:
            logger.debug(f"[Telegram desactivado] Mensaje: {text}")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error(f"Error enviando mensaje a Telegram: {exc}")
            return False

    def notify_trade_opened(
        self, symbol: str, price: float, quantity: float,
        sl: float, tp: float, mode: str, market_sentiment: str = ""
    ) -> None:
        emoji = "🟡" if mode == "paper" else "🟢"
        sentiment_map = {"bullish": "📈 Alcista", "neutral": "➡️ Neutral", "bearish": "📉 Bajista"}
        sentiment_line = (
            f"\nMercado: {sentiment_map.get(market_sentiment, market_sentiment)}"
            if market_sentiment else ""
        )
        msg = (
            f"{emoji} <b>COMPRA EJECUTADA</b> [{mode.upper()}]\n"
            f"Par: <code>{symbol}</code>\n"
            f"Precio entrada: <b>${price:,.4f}</b>\n"
            f"Cantidad: {quantity:.6f}\n"
            f"Stop Loss: ${sl:,.4f}\n"
            f"Take Profit: ${tp:,.4f}"
            f"{sentiment_line}"
        )
        self._send(msg)

    def notify_trade_closed(
        self, symbol: str, entry: float, exit_price: float,
        pnl: float, pnl_pct: float, reason: str, mode: str
    ) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        reason_map = {
            "take_profit": "Take Profit alcanzado",
            "stop_loss": "Stop Loss activado",
            "signal": "Señal de venta",
            "manual": "Cierre manual",
        }
        msg = (
            f"{emoji} <b>POSICIÓN CERRADA</b> [{mode.upper()}]\n"
            f"Par: <code>{symbol}</code>\n"
            f"Entrada: ${entry:,.4f} → Salida: ${exit_price:,.4f}\n"
            f"PnL: <b>${pnl:+.4f} ({pnl_pct:+.2f}%)</b>\n"
            f"Motivo: {reason_map.get(reason, reason)}"
        )
        self._send(msg)

    def notify_circuit_breaker(self, reason: str, value: float) -> None:
        msg = (
            f"🚨 <b>CIRCUIT BREAKER ACTIVADO</b>\n"
            f"Motivo: {reason}\n"
            f"Valor: {value:.2f}%\n"
            f"El bot ha detenido las operaciones automáticamente."
        )
        self._send(msg)

    def notify_error(self, error_msg: str) -> None:
        msg = f"⚠️ <b>ERROR EN EL BOT</b>\n<code>{error_msg[:400]}</code>"
        self._send(msg)

    def notify_status(
        self, capital: float, position_value: float,
        total_pnl: float, total_pnl_pct: float, mode: str
    ) -> None:
        emoji = "📊"
        msg = (
            f"{emoji} <b>ESTADO DEL PORTFOLIO</b>\n"
            f"Modo: {mode.upper()}\n"
            f"Capital libre: <b>${capital:,.2f}</b>\n"
            f"En posición: ${position_value:,.2f}\n"
            f"PnL total: <b>${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)</b>"
        )
        self._send(msg)

    def notify_retrain(self, accuracy: float, f1: float) -> None:
        msg = (
            f"🤖 <b>MODELO REENTRENADO</b>\n"
            f"Accuracy: {accuracy:.1%}\n"
            f"F1-Score: {f1:.3f}"
        )
        self._send(msg)
