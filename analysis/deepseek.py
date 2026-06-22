import logging
from dataclasses import dataclass
from typing import Optional

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_MODIFIER_MAP = {
    "bullish": -0.05,
    "neutral": 0.0,
    "bearish": 0.15,
}


@dataclass
class MarketContext:
    sentiment: str    # "bullish" | "neutral" | "bearish"
    modifier: float   # -0.05, 0.0, o +0.15
    reasoning: str    # explicación corta en español
    btc_price: float  # precio BTC en USD
    fear_greed: int   # Fear & Greed Index 0-100


_NEUTRAL_CONTEXT = MarketContext(
    sentiment="neutral",
    modifier=0.0,
    reasoning="Análisis no disponible — usando umbral por defecto",
    btc_price=0.0,
    fear_greed=50,
)


def _fetch_coingecko() -> dict:
    """Obtiene precio BTC/ETH, dominancia y Fear&Greed desde APIs gratuitas."""
    headers = {"Accept": "application/json"}
    result = {}

    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum&vs_currencies=usd"
            "&include_market_cap=true&include_24hr_change=true",
            headers=headers, timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        result["btc_price"] = data.get("bitcoin", {}).get("usd", 0)
        result["btc_change_24h"] = data.get("bitcoin", {}).get("usd_24h_change", 0)
        result["eth_price"] = data.get("ethereum", {}).get("usd", 0)
        result["eth_change_24h"] = data.get("ethereum", {}).get("usd_24h_change", 0)
    except Exception as e:
        logger.warning(f"CoinGecko prices no disponible: {e}")

    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/global",
            headers=headers, timeout=10,
        )
        r.raise_for_status()
        gdata = r.json().get("data", {})
        result["btc_dominance"] = round(gdata.get("market_cap_percentage", {}).get("btc", 0), 1)
        result["market_cap_change_24h"] = gdata.get("market_cap_change_percentage_24h_usd", 0)
    except Exception as e:
        logger.warning(f"CoinGecko global no disponible: {e}")

    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            headers=headers, timeout=10,
        )
        r.raise_for_status()
        fng = r.json().get("data", [{}])[0]
        result["fear_greed"] = int(fng.get("value", 50))
        result["fear_greed_label"] = fng.get("value_classification", "Neutral")
    except Exception as e:
        logger.warning(f"Fear & Greed no disponible: {e}")
        result.setdefault("fear_greed", 50)
        result.setdefault("fear_greed_label", "Neutral")

    return result


def _fetch_macro() -> dict:
    """Obtiene SP500 y DXY de los últimos 5 días via yfinance."""
    result = {}
    for ticker, key in [("^GSPC", "sp500"), ("DX-Y.NYB", "dxy")]:
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 2:
                close = df["Close"].squeeze()
                last = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                result[f"{key}_price"] = round(last, 2)
                result[f"{key}_change_pct"] = round((last / prev - 1) * 100, 2)
        except Exception as e:
            logger.warning(f"yfinance {ticker} no disponible: {e}")
    return result


def _build_prompt(crypto: dict, macro: dict) -> str:
    lines = ["=== DATOS DE MERCADO EN TIEMPO REAL ==="]

    if crypto.get("btc_price"):
        lines.append(f"BTC: ${crypto['btc_price']:,.0f} (24h: {crypto.get('btc_change_24h', 0):+.1f}%)")
    if crypto.get("eth_price"):
        lines.append(f"ETH: ${crypto['eth_price']:,.0f} (24h: {crypto.get('eth_change_24h', 0):+.1f}%)")
    if crypto.get("btc_dominance"):
        lines.append(f"Dominancia BTC: {crypto['btc_dominance']}%")
    if crypto.get("market_cap_change_24h") is not None:
        lines.append(f"Market cap cripto 24h: {crypto['market_cap_change_24h']:+.1f}%")
    if crypto.get("fear_greed") is not None:
        lines.append(f"Fear & Greed: {crypto['fear_greed']}/100 ({crypto.get('fear_greed_label', '')})")
    if macro.get("sp500_price"):
        lines.append(f"SP500: {macro['sp500_price']:,.0f} ({macro.get('sp500_change_pct', 0):+.1f}%)")
    if macro.get("dxy_price"):
        lines.append(f"DXY: {macro['dxy_price']:.2f} ({macro.get('dxy_change_pct', 0):+.1f}%)")

    lines.append(
        "\nEvalúa si el contexto es favorable para posiciones LONG en BTC/USDT. "
        "Reglas del campo modifier: bullish→-0.05, neutral→0.0, bearish→0.15. "
        "Responde SOLO con JSON válido sin markdown:\n"
        '{"sentiment":"bullish|neutral|bearish","modifier":<number>,"reasoning":"<máx 80 chars en español>"}'
    )
    return "\n".join(lines)


def _call_deepseek(prompt: str, api_key: str) -> Optional[dict]:
    import json
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai SDK no instalado — pip install openai")
        return None

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un analista cuantitativo de mercados crypto. Responde SOLO con JSON válido, sin markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=150,
            timeout=15,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        logger.warning(f"DeepSeek API error: {e}")
        return None


def _validate_response(data: dict, btc_price: float, fear_greed: int) -> MarketContext:
    sentiment = str(data.get("sentiment", "neutral")).lower()
    if sentiment not in _MODIFIER_MAP:
        sentiment = "neutral"
    modifier = _MODIFIER_MAP[sentiment]
    reasoning = str(data.get("reasoning", "Sin razonamiento"))[:100]
    return MarketContext(
        sentiment=sentiment,
        modifier=modifier,
        reasoning=reasoning,
        btc_price=btc_price,
        fear_greed=fear_greed,
    )


def get_market_context() -> MarketContext:
    """
    Obtiene contexto de mercado via DeepSeek-V3.
    Nunca lanza excepciones — retorna MarketContext neutral en caso de error.
    """
    from config.settings import DEEPSEEK_API_KEY

    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY no configurada — usando contexto neutral")
        return _NEUTRAL_CONTEXT

    try:
        crypto = _fetch_coingecko()
        macro = _fetch_macro()

        btc_price = float(crypto.get("btc_price", 0.0))
        fear_greed = int(crypto.get("fear_greed", 50))

        prompt = _build_prompt(crypto, macro)
        raw = _call_deepseek(prompt, DEEPSEEK_API_KEY)

        if raw is None:
            return _NEUTRAL_CONTEXT

        context = _validate_response(raw, btc_price, fear_greed)
        logger.info(
            f"DeepSeek: {context.sentiment.upper()} | "
            f"BTC=${btc_price:,.0f} | F&G={fear_greed} | "
            f"mod={context.modifier:+.0%} | {context.reasoning}"
        )
        return context

    except Exception as e:
        logger.error(f"Error inesperado en get_market_context: {e}", exc_info=True)
        return _NEUTRAL_CONTEXT
