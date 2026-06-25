import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Mapeo de símbolos ccxt a yfinance para datos históricos
_SYMBOL_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "BNB/USDT": "BNB-USD",
    "SOL/USDT": "SOL-USD",
    "ADA/USDT": "ADA-USD",
}


def fetch_historical_data(
    symbol: str,
    days: int = 730,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Descarga datos históricos OHLCV usando yfinance.
    Para datos de 1h descarga en chunks de 58 días (límite real de yfinance).
    """
    ticker = _SYMBOL_MAP.get(symbol, symbol.replace("/", "-").replace("USDT", "USD"))
    end = datetime.now()
    start = end - timedelta(days=days)

    logger.info(f"Descargando datos históricos: {ticker} | {days} días | intervalo {interval}")

    # yfinance limita datos de 1h y 15m a ventanas de ~60 días por request
    chunk_days = 58 if interval in ("1h", "15m", "30m", "5m") else days
    chunks = []
    chunk_end = end

    while chunk_end > start:
        chunk_start = max(chunk_end - timedelta(days=chunk_days), start)
        for attempt in range(1, 3):
            try:
                raw = yf.download(
                    ticker,
                    start=chunk_start,
                    end=chunk_end,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    multi_level_index=False,
                )
                if raw is not None and not raw.empty:
                    chunks.append(raw)
                    break
            except Exception as exc:
                logger.warning(f"Chunk fallido ({chunk_start.date()} → {chunk_end.date()}): {exc}")
                if attempt < 2:
                    time.sleep(3)

        chunk_end = chunk_start - timedelta(hours=1)
        time.sleep(0.5)

    if not chunks:
        raise RuntimeError(f"No se pudieron obtener datos históricos para {symbol}")

    df = pd.concat(chunks[::-1])  # revertir para orden cronológico
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "timestamp"
    df = df.dropna()

    logger.info(f"Descargadas {len(df)} velas para {symbol} ({len(chunks)} chunks)")
    return df


def fetch_latest_candle(
    symbol: str,
    exchange_cfg: dict,
) -> Optional[pd.DataFrame]:
    """Obtiene la última vela OHLCV en tiempo real desde Binance (testnet o live)."""
    try:
        exchange_class = getattr(ccxt, exchange_cfg.get("name", "binance"))
        exchange = exchange_class({
            "apiKey": exchange_cfg.get("api_key", ""),
            "secret": exchange_cfg.get("api_secret", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if exchange_cfg.get("testnet", True):
            exchange.set_sandbox_mode(True)

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=2)

        if not ohlcv or len(ohlcv) < 2:
            raise ValueError("El exchange no devolvió suficientes velas")

        # La última vela puede estar incompleta; usamos la penúltima (cerrada)
        row = ohlcv[-2]
        df = pd.DataFrame(
            [row],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        logger.info(f"Última vela cerrada: {df.index[-1]} | close={df['close'].iloc[-1]:.2f}")
        return df

    except ccxt.NetworkError as exc:
        logger.error(f"Error de red al conectar con el exchange: {exc}")
    except ccxt.ExchangeError as exc:
        logger.error(f"Error del exchange: {exc}")
    except Exception as exc:
        logger.error(f"Error inesperado obteniendo última vela: {exc}")

    return None


def fetch_higher_tf_bias(symbol: str) -> bool:
    """
    Retorna True si la tendencia en 4h es alcista (EMA20 > EMA50).
    En caso de error retorna True para no bloquear operaciones por fallo de datos.
    """
    ticker = _SYMBOL_MAP.get(symbol, symbol.replace("/", "-").replace("USDT", "USD"))
    try:
        raw = yf.download(
            ticker,
            period="30d",
            interval="1h",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        if raw is None or raw.empty or len(raw) < 50:
            logger.warning(f"Datos 4h insuficientes para {symbol}, bias=True por defecto")
            return True

        df_4h = raw.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

        if len(df_4h) < 50:
            return True

        close_4h = df_4h["Close"]
        ema20 = close_4h.ewm(span=20, adjust=False).mean()
        ema50 = close_4h.ewm(span=50, adjust=False).mean()

        e20 = ema20.iloc[-1]
        e50 = ema50.iloc[-1]
        diff_pct = (e20 - e50) / (e50 + 1e-10)
        # Permite operar en tendencia alcista O lateral (EMA20 no más del 1% por debajo de EMA50)
        bullish = diff_pct > -0.01
        trend_label = "ALCISTA" if e20 > e50 else ("LATERAL" if bullish else "BAJISTA")
        logger.info(
            f"Bias 4h {symbol}: EMA20={e20:.2f} vs EMA50={e50:.2f} "
            f"({diff_pct:+.2%}) → {trend_label}"
        )
        return bullish

    except Exception as exc:
        logger.warning(f"Error calculando bias 4h para {symbol}: {exc}. bias=True por defecto")
        return True
