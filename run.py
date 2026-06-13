"""
ASISTENTE TRADING — Script de control principal
================================================
Ejecuta este archivo para acceder al menú del bot.

Cómo usarlo:
    py run.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def menu():
    print("\n" + "=" * 55)
    print("       ASISTENTE TRADING — MENÚ PRINCIPAL")
    print("=" * 55)
    print("  1. Entrenar modelo de IA + Backtesting")
    print("  2. Iniciar bot de trading (loop automático)")
    print("  3. Abrir dashboard web (navegador)")
    print("  4. Ver estado del portfolio")
    print("  5. Salir")
    print("=" * 55)
    return input("  Elige una opción (1-5): ").strip()


def train_and_backtest():
    print("\n[ENTRENAMIENTO] Descargando datos y entrenando modelo...")
    print("Esto puede tardar 5-15 minutos la primera vez.\n")
    try:
        from config.settings import MODEL, RISK, TRADING
        from data.fetcher import fetch_historical_data
        from data.features import calculate_features
        from models.trainer import train_model
        from backtesting.engine import run_backtest
        from database.models import init_db

        init_db()

        print(f"  Descargando {MODEL.get('historical_days', 730)} días de {TRADING['symbol']}...")
        df = fetch_historical_data(
            symbol=TRADING["symbol"],
            days=MODEL.get("historical_days", 730),
            interval=TRADING["timeframe"],
        )
        print(f"  Datos descargados: {len(df)} velas.")

        print("  Calculando indicadores técnicos...")
        df_features = calculate_features(df, MODEL.get("target_threshold", 0.003))
        print(f"  Features listos: {len(df_features)} muestras de entrenamiento.")

        print("  Entrenando modelo de IA (puede tardar varios minutos)...")
        train_model(df_features, symbol=TRADING["symbol"])

        print("\n  Ejecutando backtesting sobre datos históricos...")
        results = run_backtest(
            df=df_features,
            initial_capital=TRADING["capital"],
            stop_loss_pct=RISK["stop_loss"],
            take_profit_pct=RISK["take_profit"],
            max_position_pct=RISK["max_position_size"],
            confidence_threshold=MODEL["prediction_threshold"],
        )

        m = results["metrics"]
        print("\n" + "=" * 50)
        print("  RESUMEN DEL BACKTESTING")
        print("=" * 50)
        print(f"  Retorno del bot:   {m['retorno_total_pct']:+.2f}%")
        print(f"  Buy & Hold (ref):  {m['buy_and_hold_pct']:+.2f}%")
        print(f"  Sharpe Ratio:      {m['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:      {m['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate:          {m['win_rate_pct']:.1f}%")
        print(f"  Operaciones:       {m['total_operaciones']}")
        print("=" * 50)
        print("  Gráfico guardado en: backtesting/results/backtest_result.png")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


def start_bot():
    print("\n[BOT] Iniciando loop automático de trading...")
    print("El bot ejecutará un ciclo cada hora.")
    print("Presiona Ctrl+C para detenerlo.\n")
    try:
        from scheduler.main import start_bot as _start
        _start()
    except KeyboardInterrupt:
        print("\nBot detenido.")


def open_dashboard():
    print("\n[DASHBOARD] Abriendo panel web en el navegador...")
    print("Cuando termine, cierra la ventana del terminal o presiona Ctrl+C.\n")
    dashboard_path = ROOT / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        cwd=str(ROOT),
    )


def show_portfolio():
    print("\n[PORTFOLIO] Estado actual:")
    try:
        from database.models import init_db
        from database.repository import (
            get_all_trades, get_latest_snapshot, get_today_pnl
        )
        from config.settings import TRADING

        init_db()
        snapshot = get_latest_snapshot()
        today_pnl = get_today_pnl()
        trades = get_all_trades()

        initial = TRADING["capital"]

        if snapshot:
            print(f"  Portfolio total:  ${snapshot.total_value:,.2f}")
            print(f"  Capital libre:    ${snapshot.cash_usd:,.2f}")
            print(f"  PnL total:        ${snapshot.total_pnl:+.2f} ({snapshot.total_pnl_pct:+.2f}%)")
            print(f"  PnL hoy:          ${today_pnl:+.4f}")
            print(f"  Drawdown actual:  {snapshot.drawdown_pct:.2f}%")
        else:
            print(f"  Sin datos aún. Capital inicial: ${initial:,.2f}")

        closed = [t for t in trades if not t.is_open]
        open_t = [t for t in trades if t.is_open]
        print(f"  Operaciones totales: {len(trades)} ({len(closed)} cerradas, {len(open_t)} abiertas)")

        if open_t:
            t = open_t[0]
            print(f"\n  POSICIÓN ABIERTA:")
            print(f"    Par: {t.symbol} | Entrada: ${t.price:.4f} | Qty: {t.quantity:.6f}")
            print(f"    Stop Loss: ${t.stop_loss:.4f} | Take Profit: ${t.take_profit:.4f}")

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    while True:
        choice = menu()
        if choice == "1":
            train_and_backtest()
        elif choice == "2":
            start_bot()
        elif choice == "3":
            open_dashboard()
        elif choice == "4":
            show_portfolio()
        elif choice == "5":
            print("\nHasta luego.\n")
            break
        else:
            print("Opción no válida. Elige entre 1 y 5.")
