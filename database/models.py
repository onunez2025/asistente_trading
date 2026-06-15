from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, create_engine
)
from sqlalchemy.orm import declarative_base

ROOT_DIR = Path(__file__).parent.parent
DB_PATH = ROOT_DIR / "database" / "trading.db"

Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class Trade(Base):
    """Registro de cada operación ejecutada (real o simulada)."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)        # "buy" o "sell"
    price = Column(Float, nullable=False)        # Precio de entrada
    quantity = Column(Float, nullable=False)     # Cantidad del activo
    value_usd = Column(Float, nullable=False)    # Valor en USD
    mode = Column(String, default="paper")       # "paper" o "live"
    stop_loss = Column(Float)
    take_profit = Column(Float)
    is_open = Column(Boolean, default=True)
    close_price = Column(Float, nullable=True)
    close_time = Column(DateTime, nullable=True)
    close_reason = Column(String, nullable=True) # "stop_loss", "take_profit", "signal", "manual"
    pnl = Column(Float, nullable=True)           # Ganancia/Pérdida en USD (después de comisiones)
    pnl_pct = Column(Float, nullable=True)       # Ganancia/Pérdida en %
    commission_paid = Column(Float, default=0.0) # Comisión total pagada en USD (compra + venta)

    def __repr__(self):
        return f"<Trade {self.side} {self.symbol} @ {self.price:.2f} | PnL: {self.pnl}>"


class Prediction(Base):
    """Registro de cada predicción del modelo."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)   # 0 = baja, 1 = sube
    probability = Column(Float, nullable=False)    # Confianza del modelo (0-1)
    action_taken = Column(Boolean, default=False)  # Si se ejecutó una orden


class PortfolioSnapshot(Base):
    """Foto del estado del portfolio cada hora."""
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cash_usd = Column(Float, nullable=False)         # Dinero en efectivo
    position_value = Column(Float, default=0.0)      # Valor de posición abierta
    total_value = Column(Float, nullable=False)      # cash + position
    daily_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    peak_value = Column(Float, nullable=False)       # Máximo histórico (para calcular drawdown)
    drawdown_pct = Column(Float, default=0.0)        # Drawdown actual en %


class ModelMetric(Base):
    """Métricas del modelo después de cada entrenamiento."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    baseline_accuracy = Column(Float)  # Accuracy de un modelo dummy (referencia)
    training_rows = Column(Integer)
    symbol = Column(String)


def init_db() -> None:
    """Crea todas las tablas si no existen."""
    Base.metadata.create_all(engine)
