import logging

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "rsi", "ema_20", "ema_50", "ema_200", "ema_cross",
    "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "atr_pct",
    "volume_ratio",
    "roc_10",
    "stoch_rsi",
    "price_vs_ema20", "price_vs_ema50",
    "candle_body", "candle_range",
]


def calculate_features(df: pd.DataFrame, target_threshold: float = 0.003) -> pd.DataFrame:
    """
    Calcula indicadores técnicos y el target para el modelo de IA.

    target = 1 si la próxima vela sube más de `target_threshold`
    target = 0 si la próxima vela baja más de `target_threshold`
    Velas con movimiento menor se eliminan (zona de ruido).
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Momentum ---
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()
    stoch = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi"] = stoch.stochrsi_k()

    # --- Tendencia ---
    df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df["ema_cross"] = (df["ema_20"] > df["ema_50"]).astype(int)
    df["price_vs_ema20"] = (close - df["ema_20"]) / df["ema_20"]
    df["price_vs_ema50"] = (close - df["ema_50"]) / df["ema_50"]

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # --- Volatilidad ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    df["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df["bb_width"] = (bb_upper - bb_lower) / (bb.bollinger_mavg() + 1e-10)

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"] = atr / (close + 1e-10)

    # --- Volumen ---
    vol_sma = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / (vol_sma + 1e-10)

    # --- Estructura de vela ---
    df["candle_body"] = (close - df["open"]) / (df["open"] + 1e-10)
    df["candle_range"] = (high - low) / (df["open"] + 1e-10)

    # --- Target ---
    future_return = close.shift(-1) / close - 1
    df["_future_return"] = future_return

    df["target"] = np.where(future_return > target_threshold, 1, 0)

    # Eliminar zona muerta (movimiento menor al threshold = señal no confiable)
    df = df[df["_future_return"].abs() > target_threshold]
    df = df.drop(columns=["_future_return"])

    df = df.dropna(subset=FEATURE_COLUMNS + ["target"])

    target_dist = df["target"].value_counts().to_dict()
    logger.info(
        f"Features calculadas | filas={len(df)} | "
        f"señales_compra={target_dist.get(1, 0)} | señales_venta={target_dist.get(0, 0)}"
    )

    return df
