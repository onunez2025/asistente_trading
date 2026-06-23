import logging
from datetime import datetime
from typing import Optional

import ccxt

from db_layer.models import Trade, init_db
from db_layer.repository import close_trade, get_open_trade, save_trade, update_trade_trailing
from risk.manager import RiskDecision

logger = logging.getLogger(__name__)


class Broker:
    """
    Capa de abstracción sobre el exchange.
    En modo paper, simula órdenes localmente y usa CCXT public API para precios en tiempo real.
    En modo live, envía órdenes reales a través de ccxt.
    """

    def __init__(self, exchange_cfg: dict, mode: str = "paper"):
        self.mode = mode
        self.exchange_cfg = exchange_cfg
        self.commission = float(exchange_cfg.get("commission", 0.001))
        self._exchange: Optional[ccxt.Exchange] = None
        self._public_exchange: Optional[ccxt.Exchange] = None

        if mode == "live":
            self._exchange = self._connect()

        self._public_exchange = self._connect_public()

        logger.info(f"Broker iniciado en modo: {mode.upper()} | comisión: {self.commission:.2%}")
        init_db()

    def _connect(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.exchange_cfg.get("name", "binance"))
        exchange = exchange_class({
            "apiKey": self.exchange_cfg.get("api_key", ""),
            "secret": self.exchange_cfg.get("api_secret", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if self.exchange_cfg.get("testnet", True):
            exchange.set_sandbox_mode(True)
        logger.info("Conectado al exchange (autenticado).")
        return exchange

    # Exchanges públicos sin restricción geográfica, en orden de prioridad
    _PUBLIC_EXCHANGES = ["kucoin", "bybit", "okx"]

    def _connect_public(self) -> Optional[ccxt.Exchange]:
        """
        Conexión pública sin API keys.
        Prueba KuCoin/Bybit/OKX primero (no tienen restricción 451 de Binance).
        """
        configured = self.exchange_cfg.get("name", "binance")
        candidates = self._PUBLIC_EXCHANGES if configured == "binance" else [configured] + self._PUBLIC_EXCHANGES
        for name in candidates:
            try:
                exchange_class = getattr(ccxt, name)
                ex = exchange_class({"enableRateLimit": True})
                logger.info(f"Exchange público conectado: {name}")
                return ex
            except Exception:
                continue
        logger.warning("No se pudo conectar a ningún exchange público. Se usará yfinance.")
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtiene el precio actual. Exchange público primero, fallback a yfinance."""
        if self._public_exchange:
            try:
                ticker = self._public_exchange.fetch_ticker(symbol)
                price = float(ticker["last"])
                logger.debug(f"Precio {self._public_exchange.id} {symbol}: {price:.4f}")
                return price
            except Exception:
                # Silencioso — fallback inmediato a yfinance
                pass

        try:
            import yfinance as yf
            from data.fetcher import _SYMBOL_MAP
            ticker = _SYMBOL_MAP.get(symbol, symbol.replace("/", "-").replace("USDT", "USD"))
            data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
            if data is not None and not data.empty:
                price = float(data["Close"].squeeze().iloc[-1])
                logger.debug(f"Precio yfinance {symbol}: {price:.4f}")
                return price
        except Exception as exc:
            logger.error(f"Precio no disponible para {symbol}: {exc}")

        return None

    def open_position(
        self,
        symbol: str,
        decision: RiskDecision,
        current_price: float,
    ) -> Optional[Trade]:
        quantity = round(decision.position_size_usd / current_price, 8)

        if self.mode == "live" and self._exchange:
            try:
                order = self._exchange.create_market_buy_order(symbol, quantity)
                fill_price = float(order.get("average", current_price))
                logger.info(f"Orden REAL enviada: compra {quantity} {symbol} @ {fill_price}")
            except Exception as exc:
                logger.error(f"Error enviando orden al exchange: {exc}")
                return None
        else:
            fill_price = current_price
            logger.info(
                f"[PAPER] Simulando compra: {quantity:.6f} {symbol} @ {fill_price:.4f} "
                f"= ${decision.position_size_usd:.2f}"
            )

        commission_open = round(decision.position_size_usd * self.commission, 6)
        logger.info(f"Comisión de compra: ${commission_open:.4f}")

        trade = Trade(
            symbol=symbol,
            side="buy",
            price=fill_price,
            quantity=quantity,
            value_usd=decision.position_size_usd,
            mode=self.mode,
            stop_loss=decision.stop_loss_price,
            take_profit=decision.take_profit_price,
            is_open=True,
            commission_paid=commission_open,
            peak_price=fill_price,
            trailing_active=False,
        )
        return save_trade(trade)

    def close_position(
        self,
        symbol: str,
        trade_id: int,
        quantity: float,
        current_price: float,
        reason: str = "signal",
    ) -> Optional[Trade]:
        if self.mode == "live" and self._exchange:
            try:
                self._exchange.create_market_sell_order(symbol, quantity)
                logger.info(f"Orden REAL enviada: venta {quantity} {symbol} @ {current_price}")
            except Exception as exc:
                logger.error(f"Error cerrando posición en exchange: {exc}")
                return None
        else:
            logger.info(
                f"[PAPER] Simulando venta: {quantity:.6f} {symbol} @ {current_price:.4f} | motivo={reason}"
            )

        return close_trade(
            trade_id=trade_id,
            close_price=current_price,
            close_reason=reason,
            commission_rate=self.commission,
        )

    def check_and_manage_open_position(
        self,
        symbol: str,
        risk_manager,
    ) -> Optional[str]:
        """
        Revisa posición abierta. Aplica trailing stop primero, luego SL/TP fijo.
        Actualiza SL y peak_price en DB si el trailing los modifica.
        """
        trade = get_open_trade(symbol)
        if not trade:
            return None

        current_price = self.get_current_price(symbol)
        if not current_price:
            return None

        peak = trade.peak_price if trade.peak_price else trade.price
        trailing_active = bool(trade.trailing_active)

        exit_reason, new_sl, new_peak, trailing_now = risk_manager.apply_trailing_stop(
            current_price=current_price,
            entry_price=trade.price,
            current_sl=trade.stop_loss,
            peak_price=peak,
            trailing_active=trailing_active,
        )

        # Persistir cambios en DB si algo cambió
        if new_sl != trade.stop_loss or new_peak != peak or trailing_now != trailing_active:
            update_trade_trailing(trade.id, new_sl, new_peak, trailing_now)

        if exit_reason:
            self.close_position(symbol, trade.id, trade.quantity, current_price, exit_reason)
            return exit_reason

        # Verificar SL/TP normales con el SL posiblemente actualizado por trailing
        exit_reason = risk_manager.check_exit_conditions(
            current_price=current_price,
            entry_price=trade.price,
            stop_loss_price=new_sl,
            take_profit_price=trade.take_profit,
        )

        if exit_reason:
            self.close_position(symbol, trade.id, trade.quantity, current_price, exit_reason)
            return exit_reason

        unrealized_pnl_pct = (current_price / trade.price - 1) * 100
        trailing_str = " | TRAILING ACTIVO" if trailing_now else ""
        logger.info(
            f"Posición abierta: {symbol} @ {trade.price:.4f} | "
            f"actual={current_price:.4f} | PnL={unrealized_pnl_pct:+.2f}% | "
            f"SL={new_sl:.4f}{trailing_str}"
        )
        return None
