import logging
from datetime import datetime
from typing import Optional

import ccxt

from database.models import Trade, init_db
from database.repository import close_trade, get_open_trade, save_trade
from risk.manager import RiskDecision

logger = logging.getLogger(__name__)


class Broker:
    """
    Capa de abstracción sobre el exchange (Binance testnet o live).
    En modo paper, simula las órdenes localmente sin enviar nada al exchange.
    En modo live, envía órdenes reales a través de ccxt.
    """

    def __init__(self, exchange_cfg: dict, mode: str = "paper"):
        self.mode = mode
        self.exchange_cfg = exchange_cfg
        self.commission = float(exchange_cfg.get("commission", 0.001))
        self._exchange: Optional[ccxt.Exchange] = None

        if mode == "live":
            self._exchange = self._connect()

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
        logger.info("Conectado al exchange.")
        return exchange

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtiene el precio actual del activo."""
        try:
            if self.mode == "paper" or self._exchange is None:
                # En paper mode sin exchange conectado, usamos yfinance como fallback
                import yfinance as yf
                from data.fetcher import _SYMBOL_MAP
                ticker = _SYMBOL_MAP.get(symbol, symbol.replace("/", "-").replace("USDT", "USD"))
                data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
                if data is not None and not data.empty:
                    return float(data["Close"].squeeze().iloc[-1])
                return None

            ticker = self._exchange.fetch_ticker(symbol)
            return float(ticker["last"])

        except Exception as exc:
            logger.error(f"Error obteniendo precio de {symbol}: {exc}")
            return None

    def open_position(
        self,
        symbol: str,
        decision: RiskDecision,
        current_price: float,
    ) -> Optional[Trade]:
        """Abre una posición de compra."""
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

        # Comisión de compra descontada del capital
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
        )
        saved = save_trade(trade)
        return saved

    def close_position(
        self,
        symbol: str,
        trade_id: int,
        quantity: float,
        current_price: float,
        reason: str = "signal",
    ) -> Optional[Trade]:
        """Cierra una posición abierta."""
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

        return close_trade(trade_id, current_price, reason, self.commission)

    def check_and_manage_open_position(
        self,
        symbol: str,
        risk_manager,
    ) -> Optional[str]:
        """
        Revisa si hay posición abierta y aplica SL/TP si corresponde.
        Retorna el motivo de cierre o None si sigue abierta.
        """
        trade = get_open_trade(symbol)
        if not trade:
            return None

        current_price = self.get_current_price(symbol)
        if not current_price:
            return None

        exit_reason = risk_manager.check_exit_conditions(
            current_price=current_price,
            entry_price=trade.price,
            stop_loss_price=trade.stop_loss,
            take_profit_price=trade.take_profit,
        )

        if exit_reason:
            self.close_position(symbol, trade.id, trade.quantity, current_price, exit_reason)
            return exit_reason

        unrealized_pnl_pct = (current_price / trade.price - 1) * 100
        logger.info(
            f"Posición abierta: {symbol} @ {trade.price:.4f} | "
            f"actual={current_price:.4f} | PnL no realizado={unrealized_pnl_pct:+.2f}%"
        )
        return None
