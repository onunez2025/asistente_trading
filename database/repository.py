import logging
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from database.models import ModelMetric, PortfolioSnapshot, Prediction, Trade, engine

logger = logging.getLogger(__name__)


def _session() -> Session:
    return Session(engine)


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
) -> Optional[Trade]:
    with _session() as s:
        trade = s.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            logger.warning(f"Trade {trade_id} no encontrado para cerrar")
            return None

        trade.close_price = close_price
        trade.close_time = datetime.utcnow()
        trade.close_reason = close_reason
        trade.is_open = False
        trade.pnl = round((close_price - trade.price) * trade.quantity, 4)
        trade.pnl_pct = round((close_price / trade.price - 1) * 100, 4)
        s.commit()
        s.refresh(trade)
        logger.info(f"Trade cerrado: PnL={trade.pnl:.2f} USD ({trade.pnl_pct:.2f}%)")
        return trade


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
