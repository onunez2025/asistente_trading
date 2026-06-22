# DeepSeek Market Analysis Integration — Design Spec

**Date:** 2026-06-22
**Status:** Approved

## Goal

Integrar DeepSeek-V3 como analizador de contexto de mercado que ajusta dinámicamente el umbral de confianza del modelo ML antes de cada decisión de trading. Mercados analizados: cripto (BTC, ETH, Fear&Greed) + macro (SP500, DXY).

## Architecture

```
trading_cycle() [cada hora]
    │
    ├─ get_market_context() → MarketContext
    │       │
    │       ├─ fetch_market_data()   (CoinGecko + yfinance)
    │       └─ call_deepseek()       (DeepSeek-V3 API)
    │
    ├─ adjusted_threshold = base_threshold + context.modifier
    │
    ├─ predict(df, threshold=adjusted_threshold)
    │
    └─ notifier.notify_*() ← incluye context.sentiment
```

**Principio de fallo seguro:** si `get_market_context()` lanza cualquier excepción (timeout, API caída, JSON inválido), retorna un `MarketContext` neutral con `modifier=0.0`. El bot nunca se interrumpe por un fallo de DeepSeek.

## Components

### `analysis/__init__.py`
Archivo vacío. Marca `analysis/` como paquete Python.

### `analysis/deepseek.py`

Responsabilidades:
1. Obtener datos de mercado en tiempo real
2. Formatear prompt para DeepSeek
3. Llamar a la API y parsear respuesta
4. Retornar `MarketContext`

**Dataclass de salida:**
```python
@dataclass
class MarketContext:
    sentiment: str          # "bullish" | "neutral" | "bearish"
    modifier: float         # ajuste al umbral: -0.05, 0.0, +0.15
    reasoning: str          # explicación en español (para Telegram)
    btc_price: float        # precio actual de BTC (informativo)
    fear_greed: int         # índice Fear & Greed 0-100
```

**Fuentes de datos (todas gratuitas y sin key):**

| Dato | Endpoint |
|---|---|
| BTC precio, ETH precio | `https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd` |
| Fear & Greed Index | `https://api.alternative.me/fng/?limit=1` |
| BTC dominancia | `https://api.coingecko.com/api/v3/global` |
| SP500 (^GSPC) | `yfinance.download("^GSPC", period="5d", interval="1d")` |
| DXY (^DXY) | `yfinance.download("DX-Y.NYB", period="5d", interval="1d")` |

Timeout por fuente: 10 segundos. Si una fuente falla, se omite su dato del prompt (el análisis continúa con los datos disponibles).

**Prompt a DeepSeek (system):**
```
You are a quantitative crypto market analyst. Given the following real-time market data,
assess the current market conditions for BTC/USDT trading.
Return ONLY a valid JSON object — no markdown, no explanation outside the JSON.
```

**Prompt a DeepSeek (user):** datos de mercado + instrucción de formato de respuesta esperada:
```json
{
  "sentiment": "bullish|neutral|bearish",
  "modifier": -0.05 | 0.0 | 0.15,
  "reasoning": "Explicación breve en español (máx 100 caracteres)"
}
```

**Reglas de modifier que DeepSeek debe seguir (incluidas en el prompt):**
- `bullish` → modifier debe ser `-0.05`
- `neutral` → modifier debe ser `0.0`
- `bearish` → modifier debe ser `+0.15`

Se valida que el JSON retornado tenga exactamente esos valores; si no, se sobrescribe con el valor correcto para el sentimiento dado.

**Parámetros de la llamada a DeepSeek:**
- Model: `deepseek-chat` (DeepSeek-V3)
- Base URL: `https://api.deepseek.com`
- Temperature: `0.1` (respuestas deterministas)
- Max tokens: `150` (la respuesta es un JSON pequeño)
- Timeout: `15` segundos

**Función pública:**
```python
def get_market_context() -> MarketContext:
    """
    Obtiene contexto de mercado via DeepSeek.
    Nunca lanza excepciones — retorna MarketContext neutral en caso de error.
    """
```

### `scheduler/main.py` — cambios

En `trading_cycle()`, después de obtener `current_price` y antes de `predict()`:

```python
# Obtener contexto de mercado via DeepSeek
from analysis.deepseek import get_market_context
context = get_market_context()
adjusted_threshold = confidence_threshold + context.modifier
logger.info(
    f"Contexto mercado: {context.sentiment.upper()} | "
    f"umbral ajustado: {adjusted_threshold:.0%} "
    f"(base={confidence_threshold:.0%}, mod={context.modifier:+.0%})"
)
```

Luego usar `adjusted_threshold` en lugar de `confidence_threshold` en la llamada a `predict()`.

### `notifications/telegram_bot.py` — cambios

Añadir `market_context: str = ""` como parámetro opcional a `notify_trade_opened()` y `notify_circuit_breaker()`. Si se provee, se añade una línea al mensaje:
```
📊 Mercado: Bajista | BTC cae junto con SP500 en rojo
```

### `.env.example` — cambio
Añadir:
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

### `config/settings.py` — cambio
Leer `DEEPSEEK_API_KEY` desde env vars y exponerla como `DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")`.

### `requirements.txt` — cambio
Añadir `openai>=1.0.0` (el SDK de OpenAI es compatible con la API de DeepSeek vía `base_url`).

## Threshold Adjustment Logic

| Sentimiento | Modifier | Umbral resultante (base 60%) | Efecto |
|---|---|---|---|
| `bullish` | −0.05 | 55% | Más trades, aprovecha tendencia alcista |
| `neutral` | 0.00 | 60% | Sin cambio |
| `bearish` | +0.15 | 75% | Menos trades, protege capital en caídas |

## Error Handling

Todos los errores se capturan en `get_market_context()` y se loggean. La función retorna:
```python
MarketContext(sentiment="neutral", modifier=0.0, reasoning="Análisis no disponible", btc_price=0.0, fear_greed=50)
```

Errores manejados:
- Timeout de CoinGecko o yfinance → datos parciales, análisis continúa
- Timeout de DeepSeek API → retorna neutral
- JSON inválido de DeepSeek → retorna neutral
- `DEEPSEEK_API_KEY` no configurada → retorna neutral con warning

## Configuration

La API key se pasa a DeepSeek vía la variable de entorno `DEEPSEEK_API_KEY`, leída en `config/settings.py`. No se hardcodea en ningún archivo. No se commitea al repo.

## Cost Estimate

- Tokens por llamada: ~800 input + ~50 output = ~850 tokens
- Costo DeepSeek-V3: ~$0.0001 por llamada
- Frecuencia: 24 llamadas/día
- **Costo diario total: ~$0.0024/día (~$0.07/mes)**

## Files Summary

| Archivo | Acción | Líneas estimadas |
|---|---|---|
| `analysis/__init__.py` | Crear | 1 |
| `analysis/deepseek.py` | Crear | ~120 |
| `scheduler/main.py` | Modificar | +15 líneas |
| `notifications/telegram_bot.py` | Modificar | +10 líneas |
| `config/settings.py` | Modificar | +3 líneas |
| `.env.example` | Modificar | +1 línea |
| `requirements.txt` | Modificar | +1 línea |
