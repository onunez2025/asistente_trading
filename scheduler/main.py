"""
Loop principal del bot. Corre cada hora automáticamente usando APScheduler.
Orquesta todos los módulos: datos → predicción → riesgo → ejecución → registro.
"""
import logging
import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import EXCHANGE, MODEL, RISK, TELEGRAM, TRADING, settings
from data.fetcher import fetch_historical_data, fetch_latest_candle
from data.features import calculate_features
from database.models import PortfolioSnapshot, init_db
from database.repository import (
    get_latest_snapshot,
    get_open_trade,
    get_today_pnl,
    save_snapshot,
)
from execution.broker import Broker
from models.predictor import is_model_trained, predict
from models.trainer import train_model
from notifications.telegram_bot import TelegramNotifier
from risk.manager import RiskManager

logger = logging.getLogger(__name__)

# ── Estado global del bot ──────────────────────────────────────────────────────
_last_retrain_date: date = date(2000, 1, 1)
_daily_reset_date: date = date(2000, 1, 1)
_initial_daily_value: float = 0.0

notifier: TelegramNotifier = None
broker: Broker = None
risk_manager: RiskManager = None


def _initialize() -> None:
    """Inicializa todos los componentes una sola vez al arrancar."""
    global notifier, broker, risk_manager, _last_retrain_date

    init_db()

    notifier = TelegramNotifier(
        token=TELEGRAM.get("token", ""),
        chat_id=TELEGRAM.get("chat_id", ""),
        enabled=TELEGRAM.get("enabled", False),
    )

    broker = Broker(
        exchange_cfg=EXCHANGE,
        mode=TRADING["mode"],
    )

    snapshot = get_latest_snapshot()
    current_capital = snapshot.total_value if snapshot else TRADING["capital"]
    saved_peak = snapshot.peak_value if snapshot else current_capital

    risk_manager = RiskManager(
        risk_cfg=RISK,
        initial_capital=current_capital,
        peak_value=saved_peak,
    )
    logger.info(f"Peak value restaurado desde BD: ${saved_peak:,.2f}")

    # Recuperar la fecha del último entrenamiento desde la BD para que
    # un reinicio del contenedor no dispare un reentrenamiento innecesario.
    from database.models import ModelMetric, engine as _engine
    from sqlalchemy.orm import Session
    with Session(_engine) as s:
        last_metric = s.query(ModelMetric).order_by(ModelMetric.timestamp.desc()).first()
        if last_metric and last_metric.timestamp:
            _last_retrain_date = last_metric.timestamp.date()
            logger.info(f"Último entrenamiento detectado en BD: {_last_retrain_date}")

    logger.info("Bot inicializado correctamente.")
    logger.info(f"Modo: {TRADING['mode'].upper()} | Par: {TRADING['symbol']} | Capital: ${current_capital:,.2f}")


def _maybe_retrain() -> None:
    """Reentrena el modelo si han pasado más de `retrain_days` días."""
    global _last_retrain_date

    days_since = (date.today() - _last_retrain_date).days
    retrain_every = MODEL.get("retrain_days", 30)

    if not is_model_trained() or days_since >= retrain_every:
        logger.info(f"Iniciando reentrenamiento del modelo (días desde último: {days_since})")
        try:
            df = fetch_historical_data(
                symbol=TRADING["symbol"],
                days=MODEL.get("historical_days", 730),
                interval=TRADING["timeframe"],
            )
            df_features = calculate_features(df, MODEL.get("target_threshold", 0.003))
            train_model(df_features, symbol=TRADING["symbol"])
            _last_retrain_date = date.today()

            from database.repository import get_latest_snapshot as _snap
            from database.models import ModelMetric
            # Notificar por Telegram (últimas métricas)
            from sqlalchemy.orm import Session
            from database.models import engine
            with Session(engine) as s:
                last_metric = s.query(ModelMetric).order_by(
                    ModelMetric.timestamp.desc()
                ).first()
                if last_metric:
                    notifier.notify_retrain(last_metric.accuracy, last_metric.f1_score)

        except Exception as exc:
            logger.error(f"Error durante el reentrenamiento: {exc}")
            notifier.notify_error(f"Error en reentrenamiento: {exc}")


def _reset_daily_state(current_value: float) -> None:
    """Resetea los contadores diarios al inicio de cada nuevo día."""
    global _daily_reset_date, _initial_daily_value

    today = date.today()
    if today != _daily_reset_date:
        _daily_reset_date = today
        _initial_daily_value = current_value
        risk_manager.reset_daily_limit()
        logger.info(f"Nuevo día iniciado. Capital de referencia: ${current_value:,.2f}")


def _save_portfolio_snapshot(
    cash: float, position_value: float, initial_capital: float
) -> None:
    snapshot = get_latest_snapshot()
    peak = snapshot.peak_value if snapshot else initial_capital
    total = cash + position_value
    pnl = total - initial_capital
    pnl_pct = pnl / initial_capital * 100
    peak = max(peak, total)
    drawdown = (peak - total) / peak * 100 if peak > 0 else 0.0

    save_snapshot(PortfolioSnapshot(
        cash_usd=cash,
        position_value=position_value,
        total_value=total,
        daily_pnl=get_today_pnl(),
        total_pnl=pnl,
        total_pnl_pct=pnl_pct,
        peak_value=peak,
        drawdown_pct=-abs(drawdown),
    ))


def trading_cycle() -> None:
    """
    Ciclo principal ejecutado cada hora:
    1. Verifica reentrenamiento
    2. Obtiene últimos datos
    3. Genera predicción
    4. Evalúa riesgo
    5. Ejecuta orden si aplica
    6. Guarda snapshot del portfolio
    """
    logger.info(f"{'='*50}")
    logger.info(f"CICLO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*50}")

    symbol = TRADING["symbol"]
    initial_capital = TRADING["capital"]
    confidence_threshold = MODEL.get("prediction_threshold", 0.60)

    try:
        # 1. Reentrenamiento si es necesario
        _maybe_retrain()

        if not is_model_trained():
            logger.warning("Modelo no entrenado. Saltando ciclo.")
            return

        # 2. Estado actual del portfolio
        snapshot = get_latest_snapshot()
        cash = snapshot.cash_usd if snapshot else initial_capital
        open_trade = get_open_trade(symbol)
        position_value = 0.0

        # 3. Precio actual
        current_price = broker.get_current_price(symbol)
        if not current_price:
            logger.warning("No se pudo obtener precio actual. Saltando ciclo.")
            return

        if open_trade:
            position_value = open_trade.quantity * current_price

        total_value = cash + position_value
        today_pnl = get_today_pnl()

        # 4. Reseteo diario y actualización del gestor de riesgo
        _reset_daily_state(total_value)
        risk_manager.update_portfolio_state(
            current_value=total_value,
            daily_pnl=today_pnl,
            initial_daily_value=_initial_daily_value,
        )

        # 5. Verificar SL/TP de posición abierta
        if open_trade:
            exit_reason = broker.check_and_manage_open_position(symbol, risk_manager)
            if exit_reason:
                closed = get_open_trade(symbol)
                if not closed:  # Se cerró correctamente
                    from database.repository import get_all_trades
                    last_trade = get_all_trades()[0]
                    notifier.notify_trade_closed(
                        symbol=symbol,
                        entry=last_trade.price,
                        exit_price=last_trade.close_price or current_price,
                        pnl=last_trade.pnl or 0,
                        pnl_pct=last_trade.pnl_pct or 0,
                        reason=exit_reason,
                        mode=TRADING["mode"],
                    )
                    cash += (last_trade.close_price or current_price) * (last_trade.quantity or 0)
                    position_value = 0.0

        # 6. Obtener datos recientes y generar predicción
        # Siempre usamos 30 días de historial (≥720 velas) para que los indicadores
        # técnicos (EMA200, RSI, MACD) se calculen correctamente en la última vela.
        df_pred = fetch_historical_data(symbol, days=30, interval="1h")
        try:
            df_live_candle = fetch_latest_candle(symbol, EXCHANGE)
            if df_live_candle is not None and not df_live_candle.empty:
                df_pred = pd.concat([df_pred, df_live_candle])
                df_pred = df_pred[~df_pred.index.duplicated(keep="last")]
        except Exception:
            pass  # el historial reciente ya tiene la última vela completa

        prediction = predict(df_pred, threshold=confidence_threshold)
        if not prediction:
            logger.warning("Predicción no disponible. Saltando ciclo.")
        else:
            logger.info(
                f"Predicción: {'COMPRAR' if prediction['signal'] == 1 else 'VENDER/ESPERAR'} "
                f"| confianza={prediction['probability']:.1%}"
            )

            # 7. Cerrar posición si el modelo predice VENDER
            if open_trade and prediction["signal"] == 0:
                closed = broker.close_position(
                    symbol=symbol,
                    trade_id=open_trade.id,
                    quantity=open_trade.quantity,
                    current_price=current_price,
                    reason="signal",
                )
                if closed:
                    notifier.notify_trade_closed(
                        symbol=symbol,
                        entry=open_trade.price,
                        exit_price=current_price,
                        pnl=closed.pnl or 0,
                        pnl_pct=closed.pnl_pct or 0,
                        reason="signal",
                        mode=TRADING["mode"],
                    )
                    cash += current_price * open_trade.quantity
                    position_value = 0.0
                    open_trade = None
                    logger.info("Posición cerrada por señal de venta del modelo.")

            # 8. Evaluar si se puede abrir posición nueva
            if not open_trade:
                decision = risk_manager.evaluate_trade(
                    signal=prediction["signal"],
                    probability=prediction["probability"],
                    confidence_threshold=confidence_threshold,
                    current_price=current_price,
                    available_capital=cash,
                    has_open_position=False,
                )

                if decision.allowed:
                    trade = broker.open_position(symbol, decision, current_price)
                    if trade:
                        cash -= decision.position_size_usd
                        position_value = decision.position_size_usd
                        notifier.notify_trade_opened(
                            symbol=symbol,
                            price=current_price,
                            quantity=trade.quantity,
                            sl=decision.stop_loss_price,
                            tp=decision.take_profit_price,
                            mode=TRADING["mode"],
                        )
                else:
                    logger.info(f"Operación bloqueada por riesgo: {decision.reason.value}")

                    if decision.reason.name in ("DAILY_LOSS_LIMIT", "MAX_DRAWDOWN"):
                        notifier.notify_circuit_breaker(
                            reason=decision.reason.value,
                            value=abs(today_pnl / initial_capital * 100),
                        )

        # 8. Guardar snapshot
        _save_portfolio_snapshot(cash, position_value, initial_capital)
        logger.info(f"Snapshot guardado | total=${cash + position_value:,.2f}")

    except Exception as exc:
        logger.error(f"Error inesperado en el ciclo de trading: {exc}", exc_info=True)
        notifier.notify_error(str(exc))


def start_bot() -> None:
    """Punto de entrada principal. Inicia el scheduler y corre el bot."""
    _initialize()

    # Ejecutar un ciclo inmediatamente al arrancar
    logger.info("Ejecutando primer ciclo inmediato...")
    trading_cycle()

    # Luego correr cada hora al minuto 0 (ej: 10:00, 11:00, 12:00...)
    scheduler = BlockingScheduler(timezone="America/Lima")
    scheduler.add_job(
        trading_cycle,
        trigger=CronTrigger(minute=0),
        id="trading_cycle",
        name="Ciclo de trading horario",
        misfire_grace_time=300,
        coalesce=True,
    )

    logger.info("Bot activo. Próxima ejecución: inicio de la siguiente hora.")
    logger.info("Presiona Ctrl+C para detener el bot.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot detenido por el usuario.")


if __name__ == "__main__":
    start_bot()
