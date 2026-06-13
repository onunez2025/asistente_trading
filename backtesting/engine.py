import logging
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from data.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "saved" / "model.pkl"
RESULTS_DIR = Path(__file__).parent.parent / "backtesting" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    max_position_pct: float = 0.15,
    confidence_threshold: float = 0.60,
) -> dict:
    """
    Simula la estrategia completa sobre datos históricos con el modelo entrenado.
    Incluye Stop Loss, Take Profit y gestión de posición.
    Devuelve métricas completas y genera gráficos.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo no encontrado. Ejecuta el entrenamiento primero."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]

    available = [f for f in features if f in df.columns]
    X = df[available].values
    probas = model.predict_proba(X)

    df = df.copy()
    df["pred_proba"] = probas[:, 1]
    df["pred_signal"] = (df["pred_proba"] >= confidence_threshold).astype(int)

    # Simulación
    capital = initial_capital
    position_qty = 0.0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    in_position = False

    equity_curve = []
    trades = []

    for i, (ts, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        equity = capital + (position_qty * price if in_position else 0)
        equity_curve.append({"timestamp": ts, "equity": equity})

        if in_position:
            # Verificar SL/TP
            exit_reason = None
            exit_price = price

            if price <= sl_price:
                exit_reason = "stop_loss"
            elif price >= tp_price:
                exit_reason = "take_profit"
            elif row["pred_signal"] == 0:
                exit_reason = "signal"

            if exit_reason:
                pnl = (exit_price - entry_price) * position_qty
                capital += exit_price * position_qty
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 4),
                    "pnl_pct": round((exit_price / entry_price - 1) * 100, 3),
                    "reason": exit_reason,
                })
                in_position = False
                position_qty = 0.0

        if not in_position and row["pred_signal"] == 1:
            position_usd = capital * max_position_pct
            if position_usd >= 10:
                position_qty = position_usd / price
                capital -= position_usd
                entry_price = price
                entry_time = ts
                sl_price = entry_price * (1 - stop_loss_pct)
                tp_price = entry_price * (1 + take_profit_pct)
                in_position = True

    # Cierra posición abierta al final del período
    if in_position:
        last_price = float(df["close"].iloc[-1])
        pnl = (last_price - entry_price) * position_qty
        capital += last_price * position_qty
        trades.append({
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "entry_price": entry_price,
            "exit_price": last_price,
            "pnl": round(pnl, 4),
            "pnl_pct": round((last_price / entry_price - 1) * 100, 3),
            "reason": "end_of_period",
        })

    # ── Métricas ──────────────────────────────────────────────────────────────
    equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
    final_value = float(equity_df["equity"].iloc[-1])
    total_return_pct = (final_value / initial_capital - 1) * 100

    buy_hold_return = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    n_trades = len(trades_df)
    win_rate = 0.0
    profit_factor = 0.0
    avg_win = 0.0
    avg_loss = 0.0

    if n_trades > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / n_trades * 100
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-10
        profit_factor = round(gross_profit / gross_loss, 3)
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

    # Sharpe Ratio (anualizado, usando retornos horarios)
    returns = equity_df["equity"].pct_change().dropna()
    sharpe = 0.0
    if returns.std() > 0:
        sharpe = round((returns.mean() / returns.std()) * np.sqrt(8760), 3)

    # Max Drawdown
    rolling_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - rolling_max) / rolling_max
    max_drawdown_pct = round(float(drawdown.min()) * 100, 2)

    metrics = {
        "capital_inicial": initial_capital,
        "capital_final": round(final_value, 2),
        "retorno_total_pct": round(total_return_pct, 2),
        "buy_and_hold_pct": round(buy_hold_return, 2),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "total_operaciones": n_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": profit_factor,
        "ganancia_promedio_usd": round(avg_win, 3),
        "perdida_promedio_usd": round(avg_loss, 3),
    }

    _log_metrics(metrics)
    _save_chart(equity_df, df, trades_df, metrics)

    return {"metrics": metrics, "equity_curve": equity_df, "trades": trades_df}


def _log_metrics(m: dict) -> None:
    logger.info("=" * 55)
    logger.info("  RESULTADOS DEL BACKTESTING")
    logger.info("=" * 55)
    logger.info(f"  Capital inicial:       ${m['capital_inicial']:>10,.2f}")
    logger.info(f"  Capital final:         ${m['capital_final']:>10,.2f}")
    logger.info(f"  Retorno total:         {m['retorno_total_pct']:>+9.2f}%")
    logger.info(f"  Buy & Hold (ref):      {m['buy_and_hold_pct']:>+9.2f}%")
    logger.info(f"  Sharpe Ratio:          {m['sharpe_ratio']:>10.3f}")
    logger.info(f"  Max Drawdown:          {m['max_drawdown_pct']:>+9.2f}%")
    logger.info(f"  Total operaciones:     {m['total_operaciones']:>10}")
    logger.info(f"  Win Rate:              {m['win_rate_pct']:>9.1f}%")
    logger.info(f"  Profit Factor:         {m['profit_factor']:>10.3f}")
    logger.info("=" * 55)


def _save_chart(
    equity_df: pd.DataFrame,
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    metrics: dict,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")

    # Panel superior: precio + entradas/salidas
    ax1.plot(price_df.index, price_df["close"], color="#4C9BE8", linewidth=0.8, label="Precio BTC")
    if not trades_df.empty:
        ax1.scatter(
            trades_df["entry_time"], trades_df["entry_price"],
            marker="^", color="#00ff88", s=60, zorder=5, label="Compra"
        )
        ax1.scatter(
            trades_df["exit_time"], trades_df["exit_price"],
            marker="v", color="#ff4444", s=60, zorder=5, label="Venta"
        )
    ax1.set_ylabel("Precio (USD)", color="white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax1.set_title("Backtesting — Señales de trading", color="white", pad=10)

    # Panel inferior: curva de equity
    ax2.plot(equity_df.index, equity_df["equity"], color="#FFD700", linewidth=1.2)
    ax2.axhline(metrics["capital_inicial"], color="#888", linestyle="--", linewidth=0.8)
    ax2.fill_between(equity_df.index, metrics["capital_inicial"], equity_df["equity"],
                     where=equity_df["equity"] >= metrics["capital_inicial"],
                     alpha=0.15, color="#00ff88")
    ax2.fill_between(equity_df.index, metrics["capital_inicial"], equity_df["equity"],
                     where=equity_df["equity"] < metrics["capital_inicial"],
                     alpha=0.15, color="#ff4444")
    ax2.set_ylabel("Portfolio (USD)", color="white")
    ax2.set_xlabel("Fecha", color="white")

    ret = metrics["retorno_total_pct"]
    color_ret = "#00ff88" if ret >= 0 else "#ff4444"
    fig.text(
        0.99, 0.01,
        f"Retorno: {ret:+.1f}%  |  Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        f"WinRate: {metrics['win_rate_pct']:.0f}%  |  MaxDD: {metrics['max_drawdown_pct']:.1f}%",
        ha="right", va="bottom", color=color_ret, fontsize=9,
        fontfamily="monospace"
    )

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30, color="white")
    plt.tight_layout()

    chart_path = RESULTS_DIR / "backtest_result.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Gráfico guardado en: {chart_path}")
