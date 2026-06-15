"""Script temporal para ejecutar entrenamiento sin menú interactivo."""
import sys
import logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from config.settings import MODEL, RISK, TRADING
from data.fetcher import fetch_historical_data
from data.features import calculate_features
from models.trainer import train_model
from backtesting.engine import run_backtest
from database.models import init_db

init_db()

print("=== PASO 1: Descargando datos de BTC/USDT ===")
df = fetch_historical_data(
    symbol=TRADING["symbol"],
    days=MODEL.get("historical_days", 90),
    interval=TRADING["timeframe"],
)
print(f"Velas descargadas: {len(df)}")
print(f"Desde: {df.index[0]}  Hasta: {df.index[-1]}")

print("\n=== PASO 2: Calculando indicadores tecnicos ===")
df_features = calculate_features(df, MODEL.get("target_threshold", 0.003))
print(f"Muestras para entrenamiento: {len(df_features)}")
feature_cols = [c for c in df_features.columns if c != "target"]
print(f"Features ({len(feature_cols)}): {feature_cols}")

print("\n=== PASO 3: Entrenando modelo de IA (puede tardar varios minutos) ===")
train_model(df_features, symbol=TRADING["symbol"])

print("\n=== PASO 4: Ejecutando backtesting ===")
results = run_backtest(
    df=df_features,
    initial_capital=TRADING["capital"],
    stop_loss_pct=RISK["stop_loss"],
    take_profit_pct=RISK["take_profit"],
    max_position_pct=RISK["max_position_size"],
    confidence_threshold=MODEL["prediction_threshold"],
)

m = results["metrics"]
print()
print("=" * 50)
print("  RESULTADOS DEL BACKTESTING")
print("=" * 50)
print(f"  Retorno del bot:   {m['retorno_total_pct']:+.2f}%")
print(f"  Buy and Hold:      {m['buy_and_hold_pct']:+.2f}%")
print(f"  Sharpe Ratio:      {m['sharpe_ratio']:.3f}")
print(f"  Max Drawdown:      {m['max_drawdown_pct']:.2f}%")
print(f"  Win Rate:          {m['win_rate_pct']:.1f}%")
print(f"  Operaciones:       {m['total_operaciones']}")
print("=" * 50)
print("Grafico guardado en: backtesting/results/backtest_result.png")
