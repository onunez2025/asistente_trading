"""
Script de demostración: puebla la base de datos con operaciones históricas
simuladas basadas en datos reales de BTC/USDT de los últimos 30 días.

Ejecutar UNA sola vez desde la terminal de EasyPanel:
    python tools/seed_demo_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datetime import datetime, timedelta
import yfinance as yf
from database.models import Trade, PortfolioSnapshot, init_db
from database.repository import save_trade, save_snapshot, close_trade
from sqlalchemy.orm import Session
from database.models import engine, Trade as TradeModel

INITIAL_CAPITAL = 1000.0
POSITION_SIZE   = 0.20   # 20% por operación
STOP_LOSS_PCT   = 0.02
TAKE_PROFIT_PCT = 0.05


def clear_existing_data():
    with Session(engine) as s:
        s.query(TradeModel).delete()
        s.query(PortfolioSnapshot).delete()
        s.commit()
    print("Base de datos limpiada.")


def seed():
    init_db()
    clear_existing_data()

    print("Descargando datos históricos de BTC...")
    df = yf.download("BTC-USD", period="30d", interval="1h",
                     progress=False, auto_adjust=True)
    if df.empty:
        print("Error: no se pudieron descargar datos.")
        return

    # Aplanar columnas multi-nivel si existen
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    prices = df["Close"].squeeze().tolist()
    timestamps = df.index.tolist()

    capital = INITIAL_CAPITAL
    trades_created = 0
    in_position = False
    entry_price = 0.0
    entry_time = None
    entry_trade_id = None
    peak = INITIAL_CAPITAL

    # Simular señales cada 6 horas (para no generar demasiadas operaciones)
    for i in range(0, len(prices) - 1, 6):
        price = float(prices[i])
        ts = timestamps[i]
        if hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()

        if not in_position:
            # Señal de compra basada en momentum simple (precio subió en últimas 3 velas)
            if i >= 3:
                momentum = (float(prices[i]) - float(prices[i-3])) / float(prices[i-3])
                if momentum > 0.001:   # subió >0.1% en 3 horas → comprar
                    position_usd = round(capital * POSITION_SIZE, 2)
                    quantity = round(position_usd / price, 8)
                    sl = round(price * (1 - STOP_LOSS_PCT), 2)
                    tp = round(price * (1 + TAKE_PROFIT_PCT), 2)

                    trade = TradeModel(
                        symbol="BTC/USDT",
                        side="buy",
                        price=price,
                        quantity=quantity,
                        value_usd=position_usd,
                        mode="paper",
                        stop_loss=sl,
                        take_profit=tp,
                        is_open=True,
                        timestamp=ts,
                    )
                    with Session(engine) as s:
                        s.add(trade)
                        s.commit()
                        s.refresh(trade)
                        entry_trade_id = trade.id

                    capital -= position_usd
                    entry_price = price
                    entry_time = ts
                    in_position = True

        else:
            # Verificar SL/TP
            sl_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
            tp_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

            close_reason = None
            close_price = price

            if price <= sl_price:
                close_reason = "stop_loss"
                close_price = sl_price
            elif price >= tp_price:
                close_reason = "take_profit"
                close_price = tp_price
            elif (ts - entry_time).total_seconds() > 48 * 3600:
                close_reason = "signal"  # forzar cierre tras 48h

            if close_reason:
                qty = round(capital * POSITION_SIZE / entry_price, 8)
                pnl = round((close_price - entry_price) * qty, 4)
                pnl_pct = round((close_price / entry_price - 1) * 100, 4)
                recovered = round(close_price * qty, 2)
                capital += recovered

                with Session(engine) as s:
                    t = s.get(TradeModel, entry_trade_id)
                    if t:
                        t.is_open = False
                        t.close_price = close_price
                        t.close_time = ts
                        t.close_reason = close_reason
                        t.pnl = pnl
                        t.pnl_pct = pnl_pct
                        s.commit()

                trades_created += 1
                in_position = False

        # Guardar snapshot de portfolio
        pos_val = 0.0
        if in_position:
            qty = round(INITIAL_CAPITAL * POSITION_SIZE / entry_price, 8)
            pos_val = round(price * qty, 2)

        total = capital + pos_val
        peak = max(peak, total)
        drawdown = -abs((peak - total) / peak * 100) if peak > 0 else 0.0

        snap = PortfolioSnapshot(
            cash_usd=capital,
            position_value=pos_val,
            total_value=total,
            daily_pnl=0.0,
            total_pnl=round(total - INITIAL_CAPITAL, 2),
            total_pnl_pct=round((total / INITIAL_CAPITAL - 1) * 100, 2),
            peak_value=peak,
            drawdown_pct=drawdown,
            timestamp=ts,
        )
        with Session(engine) as s:
            s.add(snap)
            s.commit()

    print(f"\n✅ Listo. Operaciones creadas: {trades_created}")
    print(f"   Capital final: ${capital:,.2f}")
    print(f"   PnL total: ${capital - INITIAL_CAPITAL:+,.2f}")


if __name__ == "__main__":
    seed()
