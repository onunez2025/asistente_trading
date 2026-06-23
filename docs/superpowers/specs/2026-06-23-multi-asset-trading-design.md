# Multi-Asset Trading — Design Spec
**Date:** 2026-06-23
**Status:** Approved

## Problem

The bot currently monitors only BTC/USDT. The 4h trend filter (EMA20 < EMA50) blocks all entries during bearish BTC periods, leaving the portfolio idle even when other assets may offer valid signals. Expanding to 8 assets across crypto, forex, and commodities increases signal frequency and reduces single-asset correlation risk.

---

## Goals

- Scan 8 assets every 15 minutes and enter the best signals, up to 3 simultaneous positions.
- One ML model per asset, trained independently on each asset's price history.
- Capital allocated evenly across open position slots.
- Dashboard shows per-asset signal status and all open positions in one view.
- No change to exchange connectivity, SL/TP/trailing logic, or circuit breakers.

---

## Asset Universe

| Symbol     | Class       | yfinance ticker |
|------------|-------------|-----------------|
| BTC/USDT   | Crypto      | BTC-USD         |
| ETH/USDT   | Crypto      | ETH-USD         |
| SOL/USDT   | Crypto      | SOL-USD         |
| BNB/USDT   | Crypto      | BNB-USD         |
| EUR/USD    | Forex       | EURUSD=X        |
| GBP/USD    | Forex       | GBPUSD=X        |
| XAU/USD    | Gold        | GC=F            |
| XTI/USD    | Oil (WTI)   | CL=F            |

All symbols configurable in `settings.yaml`. Adding or removing a symbol requires only a list edit — no code changes.

---

## Data Sources

- **Crypto prices (real-time):** ccxt public API (KuCoin → Bybit → OKX fallback)
- **Forex / Commodity prices (real-time):** yfinance (ccxt falls back automatically for non-crypto pairs)
- **Historical OHLCV (training + prediction):** yfinance for all assets
- **4h bias filter:** yfinance 30-day 1h data resampled to 4h — works identically for all asset classes
- **No TradingView dependency** — yfinance + ccxt covers full asset universe

---

## Architecture Changes

### 1. `settings.yaml`

`trading.symbol` (string) replaced by `trading.symbols` (list). `risk.max_open_positions` changes from `1` to `3`.

```yaml
trading:
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
    - "SOL/USDT"
    - "BNB/USDT"
    - "EUR/USD"
    - "GBP/USD"
    - "XAU/USD"
    - "XTI/USD"
  timeframe: "15m"
  capital: 1000.0
  mode: "paper"

risk:
  max_open_positions: 3
  max_position_size: 0.30   # applied per slot — see capital allocation
  ...
```

### 2. `data/fetcher.py` — `_SYMBOL_MAP`

Four new entries added to the existing map:

```python
"EUR/USD": "EURUSD=X",
"GBP/USD": "GBPUSD=X",
"XAU/USD": "GC=F",
"XTI/USD": "CL=F",
```

`fetch_higher_tf_bias(symbol)` requires no changes — yfinance handles forex and commodity tickers identically to crypto.

`fetch_latest_candle()` is crypto-only (ccxt). For non-crypto symbols it will fail silently and fall back — this is already the existing behavior for any exchange error.

### 3. `models/predictor.py` and `models/trainer.py`

Models stored as `models/model_{slug}.pkl` where slug = `symbol.replace("/", "_")`.

```python
def _model_path(symbol: str) -> Path:
    slug = symbol.replace("/", "_").replace("-", "_")
    return MODELS_DIR / f"model_{slug}.pkl"
```

All three public functions gain a `symbol` parameter:
- `is_model_trained(symbol: str) -> bool`
- `predict(df, symbol: str, threshold: float) -> dict | None`
- `train_model(df, symbol: str) -> None`

Each function resolves the model path via `_model_path(symbol)`. No other logic changes.

### 4. `db_layer/repository.py`

One new function:

```python
def get_open_trades_count() -> int:
    """Returns count of all open trades across all symbols."""
```

Existing `get_open_trade(symbol)` unchanged.

### 5. `risk/manager.py`

**New BlockReason:**
```python
MAX_POSITIONS_REACHED = "Máximo de posiciones simultáneas alcanzado"
```

**`evaluate_trade()` signature change:**
```python
def evaluate_trade(
    self,
    signal: int,
    probability: float,
    confidence_threshold: float,
    current_price: float,
    available_capital: float,
    has_open_position: bool,
    atr_value: float = 0.0,
    trend_ok: bool = True,
    open_positions_count: int = 0,
    max_open_positions: int = 1,
) -> RiskDecision:
```

Guard added before all other checks:
```python
if open_positions_count >= max_open_positions:
    return RiskDecision(allowed=False, reason=BlockReason.MAX_POSITIONS_REACHED)
```

**Capital allocation per slot:**
```python
slots_free = max(1, max_open_positions - open_positions_count)
position_usd = available_capital / slots_free
```

This distributes remaining cash evenly among free slots. With $1,000 and 3 max positions:
- 0 open → $333 per trade
- 1 open → $333 per trade (remaining $667 / 2 slots)
- 2 open → $333 per trade (remaining $333 / 1 slot)

`max_position_size` is no longer used for sizing (replaced by slot-based allocation). It remains in config for backward compatibility and may be removed in a future cleanup.

All SL/TP, trailing stop, and circuit breaker logic is unchanged.

### 6. `scheduler/main.py`

**`_initialize()`:** Reads `TRADING["symbols"]` list. `risk_manager` initialized once (global, not per-symbol).

**`trading_cycle()`:** Wraps current single-symbol logic in a `for symbol in symbols:` loop.

```
DeepSeek context — called ONCE per cycle (global signal, not per-symbol)

For each symbol in TRADING["symbols"]:
  1. Skip if model not trained for this symbol
  2. Get current price (ccxt → yfinance fallback)
  3. If no price → skip symbol
  4. If open position for this symbol → check SL/TP/trailing
  5. Fetch historical data + latest candle
  6. Calculate ATR
  7. Get 4h bias
  8. Generate ML prediction with adjusted threshold
  9. If signal == SELL and open position → close
  10. If signal == BUY and no open position:
      - get open_positions_count across ALL symbols
      - call risk_manager.evaluate_trade(..., open_positions_count, max_open_positions)
      - if allowed → open position, notify

Save portfolio snapshot once at end of cycle (global)
```

**`retrain_job()`:** Iterates over all symbols and calls `train_model(df, symbol)` for each. Sequential — one model at a time.

**`sltp_check_cycle()`:** Iterates over all symbols and calls `check_and_manage_open_position` for each open trade.

### 7. `dashboard/app.py`

**Asset grid (new):** Replaces single BTC price header. Grid of cards rendered with `st.columns(4)`, two rows of 4 assets. Each card shows:
- Symbol name and asset class badge
- Current price (real-time via broker)
- 24h change %
- ML signal: COMPRAR / ESPERAR / SIN MODELO
- Confidence % (if model trained)
- 4h bias: ALCISTA / BAJISTA

**Open positions table (extended):** Current single-position view replaced by a table with columns: Símbolo, Lado, Entrada, Precio actual, PnL%, SL, TP, Trailing.

**Metrics panel (unchanged):** Portfolio total, PnL total, PnL hoy, Efectivo libre, Drawdown — all global, no per-symbol breakdown needed.

**Equity curve, DeepSeek panel, Fear & Greed, bot stats (unchanged).**

---

## Capital Allocation Summary

| Open positions | Cash remaining | Per new trade |
|----------------|----------------|---------------|
| 0              | $1,000         | $333          |
| 1              | ~$667          | $333          |
| 2              | ~$333          | $333          |
| 3              | ~$0            | blocked       |

Maximum theoretical exposure: ~99% of capital (3 × $333). Real exposure lower due to commissions and price slippage.

---

## Error Handling

- **Symbol with no price:** Skipped silently with WARNING log. Does not block other symbols.
- **Symbol with no trained model:** Skipped with WARNING. Retrain job will produce model at 02:30 AM.
- **yfinance rate limit / timeout:** Existing retry logic in `fetch_historical_data()` handles this.
- **ccxt fails for forex/commodity pair:** Silent fallback to yfinance already in `broker.get_current_price()`.
- **All 3 slots full:** `MAX_POSITIONS_REACHED` block reason logged per symbol — no notification sent (not a circuit breaker event).

---

## First-Run Migration

1. Set `FORCE_RETRAIN=true` on first deploy to train all 8 models.
2. Existing `model.pkl` (BTC) is ignored — new path is `model_BTC_USDT.pkl`. Old file can be deleted manually.
3. `TRADING["symbol"]` references in code replaced with `TRADING["symbols"]` (list). Any leftover single-symbol reads will raise `KeyError` — caught during implementation review.

---

## Out of Scope

- Live trading on forex/commodity exchanges (paper mode only for new assets)
- Per-symbol capital allocation (all slots equal size)
- Parallel symbol scanning (sequential is sufficient for 8 assets in 15m window)
- TradingView data integration
- Per-symbol dashboard pages or drill-down views
