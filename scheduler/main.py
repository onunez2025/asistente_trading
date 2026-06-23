"""
Loop principal del bot — ciclo de 15 minutos.
Reentrenamiento en job separado a las 02:30 AM (no bloquea el ciclo de trading).
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
from data.fetcher import fetch_historical_data, fetch_latest_candle, fetch_higher_tf_bias
from data.features import calculate_features, calculate_features_inference
from db_layer.models import PortfolioSnapshot, init_db
from db_layer.repository import (
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

_last_retrain_date: date = date(2000, 1, 1)
_daily_reset_date: date = date(2000, 1, 1)
_initial_daily_value: float = 0.0

notifier: TelegramNotifier = None
broker: Broker = None
risk_manager: RiskManager = None


def _reset_portfolio_if_requested() -> None:
    """Si RESET_PORTFOLIO=true, borra snapshots y posiciones abiertas para arrancar desde cero."""
    if os.environ.get("RESET_PORTFOLIO", "").lower() not in ("true", "1", "yes"):
        return
    from db_layer.models import PortfolioSnapshot, engine as _engine
    from db_layer.repository import get_open_trade
    from sqlalchemy.orm import Session
    logger.warning("RESET_PORTFOLIO=true detectado — limpiando historial de portfolio...")
    with Session(_engine) as s:
        s.query(PortfolioSnapshot).delete()
        s.commit()
    symbol = TRADING.get("symbol", "BTC/USDT")
    open_trade = get_open_trade(symbol)
    if open_trade:
        from db_layer.repository import close_trade
        close_trade(open_trade.id, open_trade.price, "manual", 0)
        logger.warning("Posición abierta cerrada en reset.")
    logger.warning(f"Portfolio reseteado. Capital inicial: ${TRADING['capital']:,.2f}")


def _initialize() -> None:
    global notifier, broker, risk_manager, _last_retrain_date

    init_db()
    _reset_portfolio_if_requested()

    notifier = TelegramNotifier(
        token=TELEGRAM.get("token", ""),
        chat_id=TELEGRAM.get("chat_id", ""),
        enabled=TELEGRAM.get("enabled", False),
    )

    broker = Broker(exchange_cfg=EXCHANGE, mode=TRADING["mode"])

    snapshot = get_latest_snapshot()
    current_capital = snapshot.total_value if snapshot else TRADING["capital"]
    saved_peak = snapshot.peak_value if snapshot else current_capital

    risk_manager = RiskManager(
        risk_cfg=RISK,
        initial_capital=current_capital,
        peak_value=saved_peak,
    )
    logger.info(f"Peak value restaurado desde BD: ${saved_peak:,.2f}")

    from db_layer.models import ModelMetric, engine as _engine
    from sqlalchemy.orm import Session
    with Session(_engine) as s:
        last_metric = s.query(ModelMetric).order_by(ModelMetric.timestamp.desc()).first()
        if last_metric and last_metric.timestamp:
            _last_retrain_date = last_metric.timestamp.date()
            logger.info(f"Último entrenamiento detectado en BD: {_last_retrain_date}")

    # Reentrenar al arrancar si se pidió forzar
    if os.environ.get("FORCE_RETRAIN", "").lower() in ("true", "1", "yes"):
        logger.info("FORCE_RETRAIN detectado — lanzando reentrenamiento al inicio...")
        retrain_job()

    logger.info("Bot inicializado correctamente.")
    logger.info(
        f"Modo: {TRADING['mode'].upper()} | Par: {TRADING['symbol']} | "
        f"Capital: ${current_capital:,.2f} | Timeframe: {TRADING['timeframe']}"
    )


def retrain_job() -> None:
    """Reentrenamiento nocturno (02:30 AM Lima). Separado del ciclo de trading."""
    global _last_retrain_date
    logger.info("=" * 50)
    logger.info("REENTRENAMIENTO NOCTURNO INICIADO")

    days_since = (date.today() - _last_retrain_date).days
    retrain_every = MODEL.get("retrain_days", 14)
    force = os.environ.get("FORCE_RETRAIN", "").lower() in ("true", "1", "yes")

    if not force and is_model_trained() and days_since < retrain_every:
        logger.info(f"Reentrenamiento no necesario (días desde último: {days_since}/{retrain_every})")
        return
    if force:
        logger.info("FORCE_RETRAIN=true — reentrenamiento forzado")

    try:
        df = fetch_historical_data(
            symbol=TRADING["symbol"],
            days=MODEL.get("historical_days", 60),
            interval=TRADING["timeframe"],
        )
        df_features = calculate_features(df, MODEL.get("target_threshold", 0.002))
        train_model(df_features, symbol=TRADING["symbol"])
        _last_retrain_date = date.today()

        from db_layer.models import ModelMetric, engine
        from sqlalchemy.orm import Session
        with Session(engine) as s:
            last_metric = s.query(ModelMetric).order_by(ModelMetric.timestamp.desc()).first()
            if last_metric:
                notifier.notify_retrain(last_metric.accuracy, last_metric.f1_score)

        logger.info("Reentrenamiento nocturno completado.")
    except Exception as exc:
        logger.error(f"Error en reentrenamiento nocturno: {exc}", exc_info=True)
        notifier.notify_error(f"Error reentrenamiento: {exc}")


def _reset_daily_state(current_value: float) -> None:
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


def sltp_check_cycle() -> None:
    """Verifica SL/TP y trailing stop cada 5 minutos."""
    if broker is None:
        return
    symbol = TRADING["symbol"]
    try:
        if not get_open_trade(symbol):
            return
        exit_reason = broker.check_and_manage_open_position(symbol, risk_manager)
        if exit_reason:
            from db_layer.repository import get_all_trades
            trades = get_all_trades()
            if trades:
                last_trade = trades[0]
                notifier.notify_trade_closed(
                    symbol=symbol,
                    entry=last_trade.price,
                    exit_price=last_trade.close_price or 0,
                    pnl=last_trade.pnl or 0,
                    pnl_pct=last_trade.pnl_pct or 0,
                    reason=exit_reason,
                    mode=TRADING["mode"],
                )
            logger.info(f"SL/TP/Trailing ejecutado: motivo={exit_reason}")
    except Exception as exc:
        logger.error(f"Error en verificación SL/TP: {exc}", exc_info=True)


def heartbeat_job() -> None:
    """Envía señal de vida por Telegram 3 veces al día."""
    if broker is None:
        return
    try:
        snapshot = get_latest_snapshot()
        initial = TRADING["capital"]
        total = snapshot.total_value if snapshot else initial
        pct = (total / initial - 1) * 100 if initial > 0 else 0.0
        notifier.notify_heartbeat(TRADING["mode"], total, pct)
    except Exception as exc:
        logger.warning(f"Error en heartbeat: {exc}")


def trading_cycle() -> None:
    """
    Ciclo principal cada 15 minutos:
    1. Obtiene datos y extrae ATR
    2. Verifica tendencia 4h
    3. Genera predicción ML
    4. Evalúa riesgo con ATR y filtro 4h
    5. Ejecuta orden si aplica
    6. Guarda snapshot del portfolio
    """
    logger.info(f"{'─'*50}")
    logger.info(f"CICLO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'─'*50}")

    symbol = TRADING["symbol"]
    initial_capital = TRADING["capital"]
    confidence_threshold = MODEL.get("prediction_threshold", 0.65)

    try:
        if not is_model_trained():
            logger.warning("Modelo no entrenado. Saltando ciclo.")
            return

        snapshot = get_latest_snapshot()
        cash = snapshot.cash_usd if snapshot else initial_capital
        open_trade = get_open_trade(symbol)
        position_value = 0.0

        current_price = broker.get_current_price(symbol)
        if not current_price:
            logger.warning("No se pudo obtener precio actual. Saltando ciclo.")
            return

        if open_trade:
            position_value = open_trade.quantity * current_price

        total_value = cash + position_value
        today_pnl = get_today_pnl()

        # Análisis de sentimiento DeepSeek (ajusta umbral dinámicamente)
        from analysis.deepseek import get_market_context
        market_ctx = get_market_context()
        adjusted_threshold = round(confidence_threshold + market_ctx.modifier, 4)
        adjusted_threshold = max(0.55, min(0.90, adjusted_threshold))
        logger.info(
            f"Umbral ajustado: {adjusted_threshold:.0%} "
            f"(base={confidence_threshold:.0%}, mod={market_ctx.modifier:+.0%}, "
            f"mercado={market_ctx.sentiment.upper()})"
        )

        _reset_daily_state(total_value)
        risk_manager.update_portfolio_state(
            current_value=total_value,
            daily_pnl=today_pnl,
            initial_daily_value=_initial_daily_value,
        )

        # Verificar SL/TP y trailing de posición abierta
        if open_trade:
            exit_reason = broker.check_and_manage_open_position(symbol, risk_manager)
            if exit_reason:
                closed_check = get_open_trade(symbol)
                if not closed_check:
                    from db_layer.repository import get_all_trades
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
                    open_trade = None

        # Obtener datos para predicción
        df_pred = fetch_historical_data(symbol, days=30, interval=TRADING["timeframe"])
        try:
            df_live_candle = fetch_latest_candle(symbol, EXCHANGE)
            if df_live_candle is not None and not df_live_candle.empty:
                df_pred = pd.concat([df_pred, df_live_candle])
                df_pred = df_pred[~df_pred.index.duplicated(keep="last")]
        except Exception:
            pass

        # Extraer ATR absoluto para SL/TP dinámico
        atr_value = 0.0
        try:
            df_feat = calculate_features_inference(df_pred)
            if not df_feat.empty:
                atr_pct = float(df_feat["atr_pct"].iloc[-1])
                atr_value = atr_pct * current_price
                logger.info(f"ATR: {atr_pct:.4f} ({atr_value:.2f} USD)")
        except Exception as exc:
            logger.warning(f"No se pudo calcular ATR: {exc}")

        # Filtro 4h: solo abrir posiciones en tendencia alcista mayor
        trend_ok = fetch_higher_tf_bias(symbol)
        if not trend_ok:
            logger.info("Filtro 4h: mercado BAJISTA — no se abrirán nuevas posiciones")

        prediction = predict(df_pred, threshold=adjusted_threshold)
        if not prediction:
            logger.warning("Predicción no disponible. Saltando ciclo.")
        else:
            logger.info(
                f"Predicción: {'COMPRAR' if prediction['signal'] == 1 else 'VENDER/ESPERAR'} "
                f"| confianza={prediction['probability']:.1%} "
                f"| 4h={'ALCISTA' if trend_ok else 'BAJISTA'}"
            )

            # Cerrar posición si el modelo predice venta
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

            # Evaluar apertura de posición nueva con ATR y filtro 4h
            if not open_trade:
                decision = risk_manager.evaluate_trade(
                    signal=prediction["signal"],
                    probability=prediction["probability"],
                    confidence_threshold=adjusted_threshold,
                    current_price=current_price,
                    available_capital=cash,
                    has_open_position=False,
                    atr_value=atr_value,
                    trend_ok=trend_ok,
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
                            market_sentiment=market_ctx.sentiment,
                        )
                else:
                    logger.info(f"Operación bloqueada por riesgo: {decision.reason.value}")
                    if decision.reason.name in ("DAILY_LOSS_LIMIT", "MAX_DRAWDOWN"):
                        notifier.notify_circuit_breaker(
                            reason=decision.reason.value,
                            value=abs(today_pnl / initial_capital * 100),
                        )

        _save_portfolio_snapshot(cash, position_value, initial_capital)
        logger.info(f"Snapshot guardado | total=${cash + position_value:,.2f}")

    except Exception as exc:
        logger.error(f"Error inesperado en el ciclo de trading: {exc}", exc_info=True)
        notifier.notify_error(str(exc))


def start_bot() -> None:
    """Punto de entrada principal. Inicia el scheduler con todos los jobs."""
    _initialize()

    logger.info("Ejecutando primer ciclo inmediato...")
    trading_cycle()

    scheduler = BlockingScheduler(timezone="America/Lima")

    # Ciclo de trading cada 15 minutos
    scheduler.add_job(
        trading_cycle,
        trigger=CronTrigger(minute="*/15"),
        id="trading_cycle",
        name="Ciclo de trading 15m",
        misfire_grace_time=120,
        coalesce=True,
    )

    # Verificación SL/TP y trailing cada 5 minutos
    scheduler.add_job(
        sltp_check_cycle,
        trigger=CronTrigger(minute="*/5"),
        id="sltp_check",
        name="Verificación SL/TP + trailing",
        misfire_grace_time=60,
        coalesce=True,
    )

    # Reentrenamiento nocturno — job separado para no bloquear trading
    scheduler.add_job(
        retrain_job,
        trigger=CronTrigger(hour=2, minute=30),
        id="retrain",
        name="Reentrenamiento nocturno 02:30",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # Heartbeats — 3 veces al día para confirmar que el bot está vivo
    for h in [8, 14, 20]:
        scheduler.add_job(
            heartbeat_job,
            trigger=CronTrigger(hour=h, minute=0),
            id=f"heartbeat_{h}h",
            name=f"Heartbeat {h}:00",
            misfire_grace_time=300,
            coalesce=True,
        )

    logger.info("Bot activo.")
    logger.info("Ciclos: trading 15m | SL/TP+trailing 5m | Retrain 02:30 | Heartbeat 08/14/20h")
    logger.info("Presiona Ctrl+C para detener el bot.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot detenido por el usuario.")


if __name__ == "__main__":
    start_bot()
