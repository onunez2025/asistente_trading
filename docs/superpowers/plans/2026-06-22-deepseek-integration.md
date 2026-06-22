# DeepSeek Market Analysis Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrar DeepSeek-V3 como analizador de contexto de mercado (cripto + macro) que ajusta dinámicamente el umbral de confianza del ML antes de cada ciclo de trading horario.

**Architecture:** Un nuevo módulo `analysis/deepseek.py` obtiene datos reales de CoinGecko y yfinance, llama a DeepSeek-V3, y retorna un `MarketContext` con sentimiento y modificador de umbral. `trading_cycle()` en el scheduler aplica el modificador antes de llamar a `predict()`. Si DeepSeek falla por cualquier razón, el bot opera con umbral por defecto sin interrumpirse.

**Tech Stack:** Python 3.11, DeepSeek-V3 API (compatible con SDK de OpenAI), CoinGecko API (free), yfinance, APScheduler

## Global Constraints

- Model DeepSeek: `deepseek-chat` (DeepSeek-V3), base_url: `https://api.deepseek.com`
- Temperature: `0.1`, max_tokens: `150`
- Timeout DeepSeek: 15 segundos; timeout fuentes de datos: 10 segundos por fuente
- Modifier valores permitidos: `-0.05` (bullish), `0.0` (neutral), `+0.15` (bearish)
- Si `DEEPSEEK_API_KEY` vacía o API falla → retornar `MarketContext` neutral, nunca lanzar excepción
- API key en variable de entorno `DEEPSEEK_API_KEY`, nunca hardcodeada
- No hay suite de tests — verificación por output de logs y smoke test manual

---

### Task 1: Dependencias y configuración

**Files:**
- Modify: `requirements.txt:33` — añadir openai SDK
- Modify: `.env.example:22` — añadir DEEPSEEK_API_KEY
- Modify: `config/settings.py:91` — exponer DEEPSEEK_API_KEY

**Interfaces:**
- Produces: `DEEPSEEK_API_KEY` disponible como `from config.settings import DEEPSEEK_API_KEY`

- [ ] **Step 1: Añadir `openai` a `requirements.txt`**

Al final del archivo, añadir una nueva sección:

```
# === IA / ANÁLISIS DE MERCADO ===
openai>=1.0.0
```

El archivo debe quedar con estas últimas líneas:
```
# === UTILIDADES ===
pyyaml>=6.0
python-dotenv>=1.0.0

# === IA / ANÁLISIS DE MERCADO ===
openai>=1.0.0
```

- [ ] **Step 2: Añadir `DEEPSEEK_API_KEY` a `.env.example`**

Al final del archivo `.env.example`, añadir:

```
# === DEEPSEEK (análisis de mercado con IA) ===
# Obtén tu key en: platform.deepseek.com
DEEPSEEK_API_KEY=pega_tu_deepseek_api_key_aqui
```

- [ ] **Step 3: Exponer `DEEPSEEK_API_KEY` en `config/settings.py`**

Al final del archivo `config/settings.py` (después de la línea `TELEGRAM = settings["telegram"]`), añadir:

```python
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
```

- [ ] **Step 4: Verificar que la key es accesible**

```bash
python -c "from config.settings import DEEPSEEK_API_KEY; print('Key cargada:', bool(DEEPSEEK_API_KEY))"
```

Esperado: `Key cargada: True` (si tienes la key en `.env`) o `Key cargada: False` (si no, normal en CI).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env.example config/settings.py
git commit -m "feat: añadir openai SDK y DEEPSEEK_API_KEY para analisis de mercado"
```

---

### Task 2: Módulo `analysis/deepseek.py`

**Files:**
- Create: `analysis/__init__.py`
- Create: `analysis/deepseek.py`

**Interfaces:**
- Consumes: `DEEPSEEK_API_KEY` de `config.settings`
- Produces: `get_market_context() -> MarketContext` — función pública sin argumentos que nunca lanza excepción

- [ ] **Step 1: Crear `analysis/__init__.py`**

Crear el archivo vacío:
```python
```
(archivo vacío — solo marca el directorio como paquete Python)

- [ ] **Step 2: Crear `analysis/deepseek.py` completo**

```python
import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_NEUTRAL = None  # se inicializa abajo tras definir la clase


@dataclass
class MarketContext:
    sentiment: str    # "bullish" | "neutral" | "bearish"
    modifier: float   # -0.05, 0.0, o +0.15
    reasoning: str    # explicación corta en español
    btc_price: float  # precio BTC en USD
    fear_greed: int   # Fear & Greed Index 0-100


_MODIFIER_MAP = {
    "bullish": -0.05,
    "neutral": 0.0,
    "bearish": 0.15,
}

_NEUTRAL_CONTEXT = MarketContext(
    sentiment="neutral",
    modifier=0.0,
    reasoning="Análisis no disponible — usando umbral por defecto",
    btc_price=0.0,
    fear_greed=50,
)


def _fetch_coingecko() -> dict:
    """Obtiene precio BTC/ETH y dominancia desde CoinGecko (sin API key)."""
    prices_url = (
        "https://api.coingecko.com/api/v3/simple/price"
        "?ids=bitcoin,ethereum&vs_currencies=usd"
        "&include_market_cap=true&include_24hr_change=true"
    )
    global_url = "https://api.coingecko.com/api/v3/global"
    fng_url = "https://api.alternative.me/fng/?limit=1"

    result = {}
    headers = {"Accept": "application/json"}

    try:
        r = requests.get(prices_url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        result["btc_price"] = data.get("bitcoin", {}).get("usd", 0)
        result["btc_change_24h"] = data.get("bitcoin", {}).get("usd_24h_change", 0)
        result["eth_price"] = data.get("ethereum", {}).get("usd", 0)
        result["eth_change_24h"] = data.get("ethereum", {}).get("usd_24h_change", 0)
    except Exception as e:
        logger.warning(f"CoinGecko prices no disponible: {e}")

    try:
        r = requests.get(global_url, headers=headers, timeout=10)
        r.raise_for_status()
        gdata = r.json().get("data", {})
        result["btc_dominance"] = round(gdata.get("market_cap_percentage", {}).get("btc", 0), 1)
        result["total_market_cap_usd"] = gdata.get("total_market_cap", {}).get("usd", 0)
        result["market_cap_change_24h"] = gdata.get("market_cap_change_percentage_24h_usd", 0)
    except Exception as e:
        logger.warning(f"CoinGecko global no disponible: {e}")

    try:
        r = requests.get(fng_url, headers=headers, timeout=10)
        r.raise_for_status()
        fng_data = r.json().get("data", [{}])[0]
        result["fear_greed"] = int(fng_data.get("value", 50))
        result["fear_greed_label"] = fng_data.get("value_classification", "Neutral")
    except Exception as e:
        logger.warning(f"Fear & Greed no disponible: {e}")
        result["fear_greed"] = 50
        result["fear_greed_label"] = "Neutral"

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
                change_pct = (last / prev - 1) * 100
                result[f"{key}_price"] = round(last, 2)
                result[f"{key}_change_5d_pct"] = round(change_pct, 2)
        except Exception as e:
            logger.warning(f"yfinance {ticker} no disponible: {e}")

    return result


def _build_prompt(crypto: dict, macro: dict) -> str:
    lines = ["=== DATOS DE MERCADO EN TIEMPO REAL ===\n"]

    if crypto.get("btc_price"):
        lines.append(f"BTC: ${crypto['btc_price']:,.0f} (cambio 24h: {crypto.get('btc_change_24h', 0):+.1f}%)")
    if crypto.get("eth_price"):
        lines.append(f"ETH: ${crypto['eth_price']:,.0f} (cambio 24h: {crypto.get('eth_change_24h', 0):+.1f}%)")
    if crypto.get("btc_dominance"):
        lines.append(f"Dominancia BTC: {crypto['btc_dominance']}%")
    if crypto.get("market_cap_change_24h") is not None:
        lines.append(f"Market cap cripto cambio 24h: {crypto['market_cap_change_24h']:+.1f}%")
    if crypto.get("fear_greed") is not None:
        lines.append(f"Fear & Greed Index: {crypto['fear_greed']}/100 ({crypto.get('fear_greed_label', '')})")

    if macro.get("sp500_price"):
        lines.append(f"SP500: {macro['sp500_price']:,.0f} (cambio reciente: {macro.get('sp500_change_5d_pct', 0):+.1f}%)")
    if macro.get("dxy_price"):
        lines.append(f"DXY (dólar index): {macro['dxy_price']:.2f} (cambio reciente: {macro.get('dxy_change_5d_pct', 0):+.1f}%)")

    lines.append("\n=== INSTRUCCIONES ===")
    lines.append(
        "Basándote en estos datos, evalúa si el contexto es favorable para abrir posiciones LONG en BTC/USDT.\n"
        "Reglas estrictas para el campo modifier:\n"
        "  - Si sentiment es 'bullish' → modifier DEBE ser -0.05\n"
        "  - Si sentiment es 'neutral' → modifier DEBE ser 0.0\n"
        "  - Si sentiment es 'bearish' → modifier DEBE ser 0.15\n"
        "Responde ÚNICAMENTE con un JSON válido, sin markdown ni texto extra:\n"
        '{"sentiment": "bullish|neutral|bearish", "modifier": <number>, "reasoning": "<máx 80 chars en español>"}'
    )

    return "\n".join(lines)


def _call_deepseek(prompt: str, api_key: str) -> Optional[dict]:
    """Llama a DeepSeek-V3 y parsea la respuesta JSON."""
    import json
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai SDK no instalado. Ejecuta: pip install openai")
        return None

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un analista cuantitativo de mercados crypto. "
                        "Responde SOLO con JSON válido, sin markdown ni texto adicional."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=150,
            timeout=15,
        )
        raw = response.choices[0].message.content.strip()
        # Eliminar posibles bloques markdown si DeepSeek los añade
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"DeepSeek API error: {e}")
        return None


def _validate_response(data: dict) -> MarketContext:
    """Valida y normaliza la respuesta de DeepSeek."""
    sentiment = data.get("sentiment", "neutral").lower()
    if sentiment not in _MODIFIER_MAP:
        sentiment = "neutral"

    # Sobrescribir modifier con el valor canónico — no confiar en lo que diga DeepSeek
    modifier = _MODIFIER_MAP[sentiment]
    reasoning = str(data.get("reasoning", "Sin razonamiento"))[:100]

    return MarketContext(
        sentiment=sentiment,
        modifier=modifier,
        reasoning=reasoning,
        btc_price=0.0,
        fear_greed=50,
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

        btc_price = crypto.get("btc_price", 0.0)
        fear_greed = crypto.get("fear_greed", 50)

        prompt = _build_prompt(crypto, macro)
        raw = _call_deepseek(prompt, DEEPSEEK_API_KEY)

        if raw is None:
            return _NEUTRAL_CONTEXT

        context = _validate_response(raw)
        context.btc_price = btc_price
        context.fear_greed = fear_greed

        logger.info(
            f"Contexto DeepSeek: {context.sentiment.upper()} | "
            f"BTC=${btc_price:,.0f} | F&G={fear_greed} | "
            f"modifier={context.modifier:+.0%} | {context.reasoning}"
        )
        return context

    except Exception as e:
        logger.error(f"Error inesperado en get_market_context: {e}", exc_info=True)
        return _NEUTRAL_CONTEXT
```

- [ ] **Step 3: Verificar que el módulo importa sin errores**

```bash
python -c "from analysis.deepseek import get_market_context, MarketContext; print('OK')"
```
Esperado: `OK`

- [ ] **Step 4: Smoke test — llamada real a DeepSeek**

```bash
python -c "
from analysis.deepseek import get_market_context
ctx = get_market_context()
print('Sentiment:', ctx.sentiment)
print('Modifier:', ctx.modifier)
print('Reasoning:', ctx.reasoning)
print('BTC:', ctx.btc_price)
print('Fear&Greed:', ctx.fear_greed)
"
```
Esperado: imprime valores reales (sentiment = bullish/neutral/bearish, modifier = -0.05/0.0/0.15).

- [ ] **Step 5: Commit**

```bash
git add analysis/__init__.py analysis/deepseek.py
git commit -m "feat: modulo analysis/deepseek — contexto de mercado via DeepSeek-V3"
```

---

### Task 3: Integrar DeepSeek en el ciclo de trading

**Files:**
- Modify: `scheduler/main.py:208,276` — usar umbral ajustado

**Interfaces:**
- Consumes: `get_market_context() -> MarketContext` de `analysis.deepseek`
- Consumes: `confidence_threshold: float` de `MODEL.get("prediction_threshold", 0.60)`
- Produces: `adjusted_threshold` usado en `predict()` y `risk_manager.evaluate_trade()`

**Problema actual:** `predict()` en línea 276 y `risk_manager.evaluate_trade()` en línea 314 usan `confidence_threshold` fijo. Necesitan usar `adjusted_threshold`.

- [ ] **Step 1: Añadir import de `get_market_context` al inicio del bloque `try` en `trading_cycle()`**

En `scheduler/main.py`, dentro de `trading_cycle()`, después del comentario `# 3. Precio actual` y después de obtener `current_price` (línea ~228), añadir el bloque de análisis DeepSeek. Reemplazar desde:

```python
        if open_trade:
            position_value = open_trade.quantity * current_price

        total_value = cash + position_value
        today_pnl = get_today_pnl()
```

a:

```python
        if open_trade:
            position_value = open_trade.quantity * current_price

        total_value = cash + position_value
        today_pnl = get_today_pnl()

        # 3b. Contexto de mercado via DeepSeek (ajusta umbral dinámicamente)
        from analysis.deepseek import get_market_context
        market_ctx = get_market_context()
        adjusted_threshold = round(confidence_threshold + market_ctx.modifier, 4)
        adjusted_threshold = max(0.50, min(0.90, adjusted_threshold))  # clamp 50%-90%
        logger.info(
            f"Umbral ajustado: {adjusted_threshold:.0%} "
            f"(base={confidence_threshold:.0%}, mod={market_ctx.modifier:+.0%}, "
            f"mercado={market_ctx.sentiment.upper()})"
        )
```

- [ ] **Step 2: Usar `adjusted_threshold` en `predict()`**

En línea 276, cambiar:
```python
        prediction = predict(df_pred, threshold=confidence_threshold)
```
a:
```python
        prediction = predict(df_pred, threshold=adjusted_threshold)
```

- [ ] **Step 3: Usar `adjusted_threshold` en `risk_manager.evaluate_trade()`**

En la línea ~314, cambiar:
```python
                decision = risk_manager.evaluate_trade(
                    signal=prediction["signal"],
                    probability=prediction["probability"],
                    confidence_threshold=confidence_threshold,
                    current_price=current_price,
                    available_capital=cash,
                    has_open_position=False,
                )
```
a:
```python
                decision = risk_manager.evaluate_trade(
                    signal=prediction["signal"],
                    probability=prediction["probability"],
                    confidence_threshold=adjusted_threshold,
                    current_price=current_price,
                    available_capital=cash,
                    has_open_position=False,
                )
```

- [ ] **Step 4: Verificar que el scheduler parsea sin errores**

```bash
python -c "import scheduler.main; print('OK')"
```
Esperado: `OK`

- [ ] **Step 5: Commit**

```bash
git add scheduler/main.py
git commit -m "feat: umbral ML ajustado dinamicamente por contexto DeepSeek en cada ciclo"
```

---

### Task 4: Incluir sentimiento de mercado en notificaciones Telegram

**Files:**
- Modify: `notifications/telegram_bot.py:38-51` — añadir parámetro `market_sentiment`
- Modify: `scheduler/main.py:325-332` — pasar `market_ctx.sentiment` al notifier

**Interfaces:**
- Consumes: `market_ctx.sentiment: str` y `market_ctx.reasoning: str` de Task 3
- Produces: `notify_trade_opened(... market_sentiment="")` — parámetro opcional retrocompatible

- [ ] **Step 1: Modificar `notify_trade_opened` en `notifications/telegram_bot.py`**

Cambiar la firma y el cuerpo de `notify_trade_opened` de:
```python
    def notify_trade_opened(
        self, symbol: str, price: float, quantity: float,
        sl: float, tp: float, mode: str
    ) -> None:
        emoji = "🟡" if mode == "paper" else "🟢"
        msg = (
            f"{emoji} <b>COMPRA EJECUTADA</b> [{mode.upper()}]\n"
            f"Par: <code>{symbol}</code>\n"
            f"Precio entrada: <b>${price:,.4f}</b>\n"
            f"Cantidad: {quantity:.6f}\n"
            f"Stop Loss: ${sl:,.4f}\n"
            f"Take Profit: ${tp:,.4f}"
        )
        self._send(msg)
```
a:
```python
    def notify_trade_opened(
        self, symbol: str, price: float, quantity: float,
        sl: float, tp: float, mode: str, market_sentiment: str = ""
    ) -> None:
        emoji = "🟡" if mode == "paper" else "🟢"
        sentiment_map = {"bullish": "📈 Alcista", "neutral": "➡️ Neutral", "bearish": "📉 Bajista"}
        sentiment_line = (
            f"\nMercado: {sentiment_map.get(market_sentiment, market_sentiment)}"
            if market_sentiment else ""
        )
        msg = (
            f"{emoji} <b>COMPRA EJECUTADA</b> [{mode.upper()}]\n"
            f"Par: <code>{symbol}</code>\n"
            f"Precio entrada: <b>${price:,.4f}</b>\n"
            f"Cantidad: {quantity:.6f}\n"
            f"Stop Loss: ${sl:,.4f}\n"
            f"Take Profit: ${tp:,.4f}"
            f"{sentiment_line}"
        )
        self._send(msg)
```

- [ ] **Step 2: Pasar `market_sentiment` desde `scheduler/main.py`**

En `scheduler/main.py`, en la llamada a `notifier.notify_trade_opened()` (línea ~325), cambiar:
```python
                        notifier.notify_trade_opened(
                            symbol=symbol,
                            price=current_price,
                            quantity=trade.quantity,
                            sl=decision.stop_loss_price,
                            tp=decision.take_profit_price,
                            mode=TRADING["mode"],
                        )
```
a:
```python
                        notifier.notify_trade_opened(
                            symbol=symbol,
                            price=current_price,
                            quantity=trade.quantity,
                            sl=decision.stop_loss_price,
                            tp=decision.take_profit_price,
                            mode=TRADING["mode"],
                            market_sentiment=market_ctx.sentiment,
                        )
```

- [ ] **Step 3: Verificar que el módulo de notificaciones importa sin errores**

```bash
python -c "from notifications.telegram_bot import TelegramNotifier; print('OK')"
```
Esperado: `OK`

- [ ] **Step 4: Commit**

```bash
git add notifications/telegram_bot.py scheduler/main.py
git commit -m "feat: incluir sentimiento DeepSeek en notificacion de compra Telegram"
```

---

### Task 5: Deploy a EasyPanel

- [ ] **Step 1: Añadir `DEEPSEEK_API_KEY` en las variables de entorno de EasyPanel**

En el panel de EasyPanel → servicio `bot` → pestaña **Environment** → añadir:
```
DEEPSEEK_API_KEY=tu_deepseek_api_key_aqui
```

- [ ] **Step 2: Push a GitHub y disparar webhook**

```bash
git push origin main
curl -X POST "http://72.61.75.5:3000/api/deploy/6a4675f7df410b5f9905abc0e3bbc34d276879ecf576e25d"
```

- [ ] **Step 3: Verificar en logs de EasyPanel**

En los logs del contenedor `bot`, en el primer ciclo deberías ver líneas como:
```
Contexto DeepSeek: NEUTRAL | BTC=$105,000 | F&G=62 | modifier=+0% | Mercado en equilibrio
Umbral ajustado: 60% (base=60%, mod=+0%, mercado=NEUTRAL)
```

Si el sentimiento es bajista:
```
Contexto DeepSeek: BEARISH | BTC=$98,000 | F&G=28 | modifier=+15% | BTC cae junto al SP500
Umbral ajustado: 75% (base=60%, mod=+15%, mercado=BEARISH)
```
