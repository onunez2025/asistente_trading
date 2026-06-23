import logging
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from db_layer.models import ModelMetric, PortfolioSnapshot, Prediction, Trade, engine

logger = logging.getLogger(__name__)

__version__ = "1.1.0"


def _session() -> Session:
    return Session(engine, expire_on_commit=False)


# ── Trades ──────────────────────────────────────────────────────────────────

def save_trade(trade: Trade) -> Trade:
    with _session() as s:
        s.add(trade)
        s.commit()
        s.refresh(trade)
        logger.info(f"Trade guardado: {trade}")
        return trade


def get_open_trade(symbol: str) -> Optional[Trade]:
    with _session() as s:
        return (
            s.query(Trade)
            .filter(Trade.symbol == symbol, Trade.is_open == True)
            .first()
        )


def close_trade(
    trade_id: int,
    close_price: float,
    close_reason: str = "signal",
    commission_rate: float = None,
) -> Optional[Trade]:
    if commission_rate is None:
        from config.settings import EXCHANGE
        commission_rate = float(EXCHANGE.get("commission", 0.001))

    with _session() as s:
        trade = s.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            logger.warning(f"Trade {trade_id} no encontrado para cerrar")
            return None

        # Comisión de compra (ya pagada al abrir) + comisión de venta (se paga ahora)
        commission_open  = trade.value_usd * commission_rate
        commission_close = close_price * trade.quantity * commission_rate
        total_commission = round(commission_open + commission_close, 6)

        gross_pnl = (close_price - trade.price) * trade.quantity
        net_pnl   = gross_pnl - total_commission

        trade.close_price     = close_price
        trade.close_time      = datetime.utcnow()
        trade.close_reason    = close_reason
        trade.is_open         = False
        trade.commission_paid = total_commission
        trade.pnl             = round(net_pnl, 4)
        trade.pnl_pct         = round((close_price / trade.price - 1) * 100, 4)
        s.commit()
        s.refresh(trade)
        logger.info(
            f"Trade cerrado: PnL bruto={gross_pnl:.4f} | "
            f"comisión={total_commission:.4f} | PnL neto={net_pnl:.4f} USD"
        )
        return trade


def update_trade_trailing(
    trade_id: int,
    new_sl: float,
    new_peak: float,
    trailing_active: bool,
) -> None:
    """Actualiza SL, precio pico y estado de trailing de una posición abierta."""
    with _session() as s:
        trade = s.query(Trade).filter(Trade.id == trade_id).first()
        if trade:
            trade.stop_loss = round(new_sl, 6)
            trade.peak_price = round(new_peak, 6)
            trade.trailing_active = trailing_active
            s.commit()


def get_all_trades() -> List[Trade]:
    with _session() as s:
        return s.query(Trade).order_by(Trade.timestamp.desc()).all()


def get_today_pnl() -> float:
    with _session() as s:
        today_start = datetime.combine(date.today(), datetime.min.time())
        trades = (
            s.query(Trade)
            .filter(Trade.close_time >= today_start, Trade.is_open == False)
            .all()
        )
        return round(sum(t.pnl or 0.0 for t in trades), 4)


# ── Predictions ──────────────────────────────────────────────────────────────

def save_prediction(prediction: Prediction) -> Prediction:
    with _session() as s:
        s.add(prediction)
        s.commit()
        s.refresh(prediction)
        return prediction


# ── Portfolio Snapshots ───────────────────────────────────────────────────────

def save_snapshot(snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
    with _session() as s:
        s.add(snapshot)
        s.commit()
        s.refresh(snapshot)
        return snapshot


def get_latest_snapshot() -> Optional[PortfolioSnapshot]:
    with _session() as s:
        return (
            s.query(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .first()
        )


def get_snapshots_last_days(days: int = 30) -> List[PortfolioSnapshot]:
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=days)
    with _session() as s:
        return (
            s.query(PortfolioSnapshot)
            .filter(PortfolioSnapshot.timestamp >= cutoff)
            .order_by(PortfolioSnapshot.timestamp.asc())
            .all()
        )


# ── Model Metrics ─────────────────────────────────────────────────────────────

def save_model_metric(metric: ModelMetric) -> ModelMetric:
    with _session() as s:
        s.add(metric)
        s.commit()
        s.refresh(metric)
        logger.info(
            f"Métricas del modelo guardadas | accuracy={metric.accuracy:.3f} | "
            f"f1={metric.f1_score:.3f} | baseline={metric.baseline_accuracy:.3f}"
        )
        return metric


def get_latest_model_metric() -> Optional[ModelMetric]:
    with _session() as s:
        return (
            s.query(ModelMetric)
            .order_by(ModelMetric.timestamp.desc())
            .first()
        )
