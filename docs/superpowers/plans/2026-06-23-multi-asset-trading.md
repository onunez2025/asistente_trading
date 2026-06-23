# Multi-Asset Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the bot from BTC/USDT-only to 8 assets (crypto, forex, commodities) with up to 3 simultaneous positions and one ML model per asset.

**Architecture:** The existing single-symbol trading cycle is wrapped in a `for symbol in symbols:` loop. Each asset gets its own `model_{slug}.pkl` file. The risk manager receives the global open-position count and allocates capital evenly across free slots. The dashboard gains a per-asset signal grid and a multi-row positions table.

**Tech Stack:** Python 3.11, LightGBM, yfinance, ccxt, SQLAlchemy, Streamlit, pytest

## Global Constraints

- `settings.yaml` `trading.symbols` is the single source of truth for the asset list — no hardcoded lists anywhere in Python code.
- Model files live in `models/saved/model_{slug}.pkl` where slug = `symbol.replace("/", "_")`.
- All new functions must handle `KeyError`/`AttributeError` from missing config keys gracefully.
- Paper mode only — no live exchange orders for forex/commodity symbols.
- Test files go in `tests/` at project root. Run with: `pytest tests/ -v` from project root.
- Python path: tests must add project root to `sys.path` — use the fixture in Task 1.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `settings.yaml` | Modify | `symbol` → `symbols` list; `max_open_positions: 3` |
| `data/fetcher.py` | Modify | Extend `_SYMBOL_MAP` with forex/commodity tickers |
| `models/predictor.py` | Modify | Add `_model_path(symbol)`, `MODELS_DIR`; add `symbol` param to all public functions |
| `models/trainer.py` | Modify | Import `_model_path` from predictor; use per-symbol path instead of hardcoded `MODEL_PATH` |
| `db_layer/repository.py` | Modify | Add `get_open_trades_count()` and `get_model_metric_by_symbol(symbol)` |
| `risk/manager.py` | Modify | Add `MAX_POSITIONS_REACHED`; update `evaluate_trade()` with slot-based capital allocation |
| `scheduler/main.py` | Modify | Loop `trading_cycle()`, `retrain_job()`, `sltp_check_cycle()` over all symbols |
| `dashboard/app.py` | Modify | Per-asset signal grid; multi-row open positions table; model count in header |
| `tests/conftest.py` | Create | Shared pytest fixtures (sys.path, minimal risk_cfg) |
| `tests/test_symbol_map.py` | Create | Verify all 8 symbols resolve in `_SYMBOL_MAP` |
| `tests/test_model_paths.py` | Create | Verify `_model_path` slug generation and `is_model_trained` per-symbol |
| `tests/test_repository.py` | Create | Verify `get_open_trades_count()` and `get_model_metric_by_symbol()` |
| `tests/test_risk_manager.py` | Create | Verify `MAX_POSITIONS_REACHED` and slot-based capital allocation |

---

## Task 1: Config & Symbol Map

**Files:**
- Modify: `settings.yaml`
- Modify: `data/fetcher.py`
- Create: `tests/conftest.py`
- Create: `tests/test_symbol_map.py`

**Interfaces:**
- Produces: `TRADING["symbols"]` (list of str) — consumed by Tasks 5, 6
- Produces: `_SYMBOL_MAP` with 8 entries — consumed by Tasks 5, 6

- [ ] **Step 1: Write the failing test**

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

# Add project root to path so all imports work from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
```

Create `tests/test_symbol_map.py`:
```python
from data.fetcher import _SYMBOL_MAP

EXPECTED_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "EUR/USD", "GBP/USD", "XAU/USD", "XTI/USD",
]

def test_symbol_map_covers_all_assets():
    for sym in EXPECTED_SYMBOLS:
        assert sym in _SYMBOL_MAP, f"{sym} missing from _SYMBOL_MAP"

def test_forex_tickers_use_yfinance_format():
    assert _SYMBOL_MAP["EUR/USD"] == "EURUSD=X"
    assert _SYMBOL_MAP["GBP/USD"] == "GBPUSD=X"

def test_commodity_tickers_use_yfinance_format():
    assert _SYMBOL_MAP["XAU/USD"] == "GC=F"
    assert _SYMBOL_MAP["XTI/USD"] == "CL=F"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_symbol_map.py -v
```
Expected: FAIL — `EUR/USD`, `GBP/USD`, `XAU/USD`, `XTI/USD` missing from `_SYMBOL_MAP`.

- [ ] **Step 3: Update `data/fetcher.py` — extend `_SYMBOL_MAP`**

Replace the existing `_SYMBOL_MAP` dict (lines 13–19) with:
```python
_SYMBOL_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "BNB/USDT": "BNB-USD",
    "SOL/USDT": "SOL-USD",
    "ADA/USDT": "ADA-USD",
    "EUR/USD":  "EURUSD=X",
    "GBP/USD":  "GBPUSD=X",
    "XAU/USD":  "GC=F",
    "XTI/USD":  "CL=F",
}
```

- [ ] **Step 4: Update `settings.yaml` — replace `symbol` with `symbols` list**

Replace the `trading:` block:
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
```

Also update `risk.max_open_positions`:
```yaml
risk:
  max_open_positions: 3
  max_position_size: 0.30
  stop_loss: 0.02
  take_profit: 0.04
  atr_sl_multiplier: 1.5
  atr_tp_multiplier: 3.0
  trailing_breakeven_pct: 1.5
  trailing_activate_pct: 3.0
  trailing_distance_pct: 1.5
  daily_loss_limit: 0.05
  max_drawdown: 0.20
  max_open_positions: 3
```

- [ ] **Step 5: Run test to verify it passes**

```
pytest tests/test_symbol_map.py -v
```
Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add settings.yaml data/fetcher.py tests/conftest.py tests/test_symbol_map.py
git commit -m "feat: extend _SYMBOL_MAP to 8 assets; settings.yaml symbols list"
```

---

## Task 2: Per-Symbol Model Paths

**Files:**
- Modify: `models/predictor.py`
- Modify: `models/trainer.py`
- Create: `tests/test_model_paths.py`

**Interfaces:**
- Produces: `_model_path(symbol: str) -> Path` — used internally by predictor and trainer
- Produces: `MODELS_DIR: Path` — exported for tests and trainer import
- Produces: `is_model_trained(symbol: str) -> bool`
- Produces: `predict(df_raw: pd.DataFrame, symbol: str, threshold: float = 0.60) -> Optional[dict]`
- Consumes: nothing from other tasks

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_paths.py`:
```python
import sys
from pathlib import Path
import pytest

def test_model_path_btc():
    from models.predictor import _model_path
    p = _model_path("BTC/USDT")
    assert p.name == "model_BTC_USDT.pkl"

def test_model_path_forex():
    from models.predictor import _model_path
    p = _model_path("EUR/USD")
    assert p.name == "model_EUR_USD.pkl"

def test_model_path_commodity():
    from models.predictor import _model_path
    p = _model_path("XAU/USD")
    assert p.name == "model_XAU_USD.pkl"

def test_is_model_trained_false_when_file_missing(tmp_path, monkeypatch):
    import models.predictor as pred_mod
    monkeypatch.setattr(pred_mod, "MODELS_DIR", tmp_path)
    assert pred_mod.is_model_trained("BTC/USDT") is False

def test_is_model_trained_true_when_file_exists(tmp_path, monkeypatch):
    import models.predictor as pred_mod
    monkeypatch.setattr(pred_mod, "MODELS_DIR", tmp_path)
    (tmp_path / "model_BTC_USDT.pkl").touch()
    assert pred_mod.is_model_trained("BTC/USDT") is True
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_model_paths.py -v
```
Expected: FAIL — `_model_path` and `MODELS_DIR` not defined, `is_model_trained` missing `symbol` param.

- [ ] **Step 3: Rewrite `models/predictor.py`**

Replace the entire file content:
```python
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from data.features import FEATURE_COLUMNS, calculate_features_inference

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(symbol: str) -> Path:
    slug = symbol.replace("/", "_")
    return MODELS_DIR / f"model_{slug}.pkl"


def is_model_trained(symbol: str) -> bool:
    return _model_path(symbol).exists()


def predict(df_raw: pd.DataFrame, symbol: str, threshold: float = 0.60) -> Optional[dict]:
    """
    Recibe datos OHLCV crudos, calcula features y devuelve la predicción.

    Retorna:
        {
            "signal": 1 (comprar) o 0 (vender/esperar),
            "probability": float entre 0 y 1,
            "confidence_ok": bool (True si supera el umbral mínimo)
        }
        None si el modelo no está entrenado o hay un error.
    """
    if not is_model_trained(symbol):
        logger.error(f"Modelo no entrenado para {symbol}. Ejecuta el entrenamiento primero.")
        return None

    try:
        artifact = joblib.load(_model_path(symbol))
        model = artifact["model"]
        features = artifact["features"]

        df_features = calculate_features_inference(df_raw)
        if df_features.empty:
            logger.warning("No hay suficientes datos para calcular features.")
            return None

        last_row = df_features[features].iloc[[-1]]
        proba = model.predict_proba(last_row)[0]
        buy_proba = float(proba[1])
        signal = 1 if buy_proba >= threshold else 0

        result = {
            "signal": signal,
            "probability": buy_proba,
            "confidence_ok": buy_proba >= threshold,
        }

        action = "COMPRAR" if signal == 1 else "VENDER/ESPERAR"
        status = "OK" if result["confidence_ok"] else "IGNORADA (baja confianza)"
        logger.info(
            f"[{symbol}] Predicción: {action} | confianza={buy_proba:.1%} | {status}"
        )
        return result

    except Exception as exc:
        logger.error(f"[{symbol}] Error al generar predicción: {exc}")
        return None
```

- [ ] **Step 4: Update `models/trainer.py` — use `_model_path(symbol)`**

At the top of `trainer.py`, replace lines 23–24:
```python
MODEL_PATH = Path(__file__).parent / "saved" / "model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
```
with:
```python
from models.predictor import MODELS_DIR, _model_path
MODELS_DIR.mkdir(parents=True, exist_ok=True)
```

At line 132, replace:
```python
joblib.dump({"model": final_model, "features": available}, MODEL_PATH)
logger.info(f"Modelo guardado en {MODEL_PATH}")
```
with:
```python
path = _model_path(symbol)
joblib.dump({"model": final_model, "features": available}, path)
logger.info(f"Modelo guardado en {path}")
```

Also remove the unused `from pathlib import Path` import in trainer.py if `Path` is no longer used (check if it's still referenced elsewhere in the file — if not, remove it).

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_model_paths.py -v
```
Expected: 5 PASSED.

- [ ] **Step 6: Commit**

```bash
git add models/predictor.py models/trainer.py tests/test_model_paths.py
git commit -m "feat: per-symbol model paths — model_{slug}.pkl"
```

---

## Task 3: Repository — `get_open_trades_count` and `get_model_metric_by_symbol`

**Files:**
- Modify: `db_layer/repository.py`
- Create: `tests/test_repository.py`

**Interfaces:**
- Produces: `get_open_trades_count() -> int` — consumed by Task 5 (scheduler)
- Produces: `get_model_metric_by_symbol(symbol: str) -> Optional[ModelMetric]` — consumed by Task 6 (dashboard)
- Consumes: existing `Trade`, `ModelMetric` models — no schema changes needed

- [ ] **Step 1: Write failing tests**

Create `tests/test_repository.py`:
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db_layer.models import Base, Trade, ModelMetric
from db_layer import repository


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    """Swap the module-level engine for an in-memory SQLite for each test."""
    import db_layer.models as models_mod
    import db_layer.repository as repo_mod

    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)

    monkeypatch.setattr(models_mod, "engine", test_engine)
    monkeypatch.setattr(repo_mod, "engine", test_engine)
    yield test_engine


def _make_open_trade(symbol: str) -> Trade:
    return Trade(symbol=symbol, side="buy", price=100.0, quantity=1.0,
                 value_usd=100.0, mode="paper", is_open=True)


def test_get_open_trades_count_zero():
    assert repository.get_open_trades_count() == 0


def test_get_open_trades_count_one(in_memory_db):
    with Session(in_memory_db) as s:
        s.add(_make_open_trade("BTC/USDT"))
        s.commit()
    assert repository.get_open_trades_count() == 1


def test_get_open_trades_count_multiple_symbols(in_memory_db):
    with Session(in_memory_db) as s:
        s.add(_make_open_trade("BTC/USDT"))
        s.add(_make_open_trade("ETH/USDT"))
        s.commit()
    assert repository.get_open_trades_count() == 2


def test_closed_trades_not_counted(in_memory_db):
    with Session(in_memory_db) as s:
        t = _make_open_trade("BTC/USDT")
        t.is_open = False
        s.add(t)
        s.commit()
    assert repository.get_open_trades_count() == 0


def test_get_model_metric_by_symbol_none_when_empty():
    result = repository.get_model_metric_by_symbol("BTC/USDT")
    assert result is None


def test_get_model_metric_by_symbol_returns_correct(in_memory_db):
    from datetime import datetime
    with Session(in_memory_db) as s:
        s.add(ModelMetric(symbol="BTC/USDT", accuracy=0.55, f1_score=0.48,
                          precision=0.50, recall=0.46, baseline_accuracy=0.50,
                          training_rows=1000, timestamp=datetime(2026, 1, 1)))
        s.add(ModelMetric(symbol="ETH/USDT", accuracy=0.60, f1_score=0.52,
                          precision=0.55, recall=0.50, baseline_accuracy=0.50,
                          training_rows=1000, timestamp=datetime(2026, 1, 2)))
        s.commit()
    result = repository.get_model_metric_by_symbol("ETH/USDT")
    assert result is not None
    assert abs(result.accuracy - 0.60) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_repository.py -v
```
Expected: FAIL — `get_open_trades_count` and `get_model_metric_by_symbol` not defined.

- [ ] **Step 3: Add functions to `db_layer/repository.py`**

After the `get_today_pnl()` function (around line 107), add:

```python
def get_open_trades_count() -> int:
    """Returns count of all open trades across all symbols."""
    with _session() as s:
        return s.query(Trade).filter(Trade.is_open == True).count()
```

After `get_latest_model_metric()` (around line 171), add:

```python
def get_model_metric_by_symbol(symbol: str) -> Optional[ModelMetric]:
    with _session() as s:
        return (
            s.query(ModelMetric)
            .filter(ModelMetric.symbol == symbol)
            .order_by(ModelMetric.timestamp.desc())
            .first()
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_repository.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add db_layer/repository.py tests/test_repository.py
git commit -m "feat: add get_open_trades_count() and get_model_metric_by_symbol()"
```

---

## Task 4: Risk Manager — Multi-Position Support

**Files:**
- Modify: `risk/manager.py`
- Create: `tests/test_risk_manager.py`

**Interfaces:**
- Produces: `BlockReason.MAX_POSITIONS_REACHED`
- Produces: `evaluate_trade(..., open_positions_count: int = 0, max_open_positions: int = 1) -> RiskDecision`
  - Capital allocation: `position_usd = available_capital / max(1, max_open_positions - open_positions_count)`
- Consumes: nothing from other tasks

- [ ] **Step 1: Write failing tests**

Create `tests/test_risk_manager.py`:
```python
import pytest
from risk.manager import BlockReason, RiskDecision, RiskManager

RISK_CFG = {
    "max_position_size": 0.30,
    "stop_loss": 0.02,
    "take_profit": 0.04,
    "daily_loss_limit": 0.05,
    "max_drawdown": 0.20,
    "atr_sl_multiplier": 0.0,
    "atr_tp_multiplier": 0.0,
    "trailing_breakeven_pct": 1.5,
    "trailing_activate_pct": 3.0,
    "trailing_distance_pct": 1.5,
}


def _rm() -> RiskManager:
    return RiskManager(risk_cfg=RISK_CFG, initial_capital=1000.0)


def test_max_positions_reached_blocks_trade():
    rm = _rm()
    decision = rm.evaluate_trade(
        signal=1, probability=0.85, confidence_threshold=0.65,
        current_price=100.0, available_capital=0.0,
        has_open_position=False,
        open_positions_count=3, max_open_positions=3,
    )
    assert decision.allowed is False
    assert decision.reason == BlockReason.MAX_POSITIONS_REACHED


def test_slot_allocation_zero_open():
    rm = _rm()
    decision = rm.evaluate_trade(
        signal=1, probability=0.85, confidence_threshold=0.65,
        current_price=100.0, available_capital=1000.0,
        has_open_position=False,
        open_positions_count=0, max_open_positions=3,
    )
    assert decision.allowed is True
    assert abs(decision.position_size_usd - 333.33) < 1.0  # 1000 / 3


def test_slot_allocation_one_open():
    rm = _rm()
    decision = rm.evaluate_trade(
        signal=1, probability=0.85, confidence_threshold=0.65,
        current_price=100.0, available_capital=667.0,
        has_open_position=False,
        open_positions_count=1, max_open_positions=3,
    )
    assert decision.allowed is True
    assert abs(decision.position_size_usd - 333.5) < 1.0  # 667 / 2


def test_slot_allocation_two_open():
    rm = _rm()
    decision = rm.evaluate_trade(
        signal=1, probability=0.85, confidence_threshold=0.65,
        current_price=100.0, available_capital=333.0,
        has_open_position=False,
        open_positions_count=2, max_open_positions=3,
    )
    assert decision.allowed is True
    assert abs(decision.position_size_usd - 333.0) < 1.0  # 333 / 1


def test_existing_blocking_reasons_still_work():
    rm = _rm()
    rm._drawdown_triggered = True
    decision = rm.evaluate_trade(
        signal=1, probability=0.85, confidence_threshold=0.65,
        current_price=100.0, available_capital=1000.0,
        has_open_position=False,
        open_positions_count=0, max_open_positions=3,
    )
    assert decision.reason == BlockReason.MAX_DRAWDOWN


def test_low_confidence_blocks_even_with_slots_free():
    rm = _rm()
    decision = rm.evaluate_trade(
        signal=1, probability=0.50, confidence_threshold=0.65,
        current_price=100.0, available_capital=1000.0,
        has_open_position=False,
        open_positions_count=0, max_open_positions=3,
    )
    assert decision.allowed is False
    assert decision.reason == BlockReason.LOW_CONFIDENCE
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_risk_manager.py -v
```
Expected: FAIL — `MAX_POSITIONS_REACHED` not in `BlockReason`, `evaluate_trade` missing new params.

- [ ] **Step 3: Update `risk/manager.py`**

Add `MAX_POSITIONS_REACHED` to the `BlockReason` enum (after the existing entries):
```python
class BlockReason(Enum):
    DAILY_LOSS_LIMIT = "Límite de pérdida diaria alcanzado"
    MAX_DRAWDOWN = "Circuit breaker: drawdown máximo alcanzado"
    POSITION_OPEN = "Ya hay una posición abierta"
    LOW_CONFIDENCE = "Confianza del modelo insuficiente"
    INSUFFICIENT_CAPITAL = "Capital insuficiente para operar"
    TREND_FILTER = "Tendencia 4h bajista — no se abre posición"
    MAX_POSITIONS_REACHED = "Máximo de posiciones simultáneas alcanzado"
    OK = "OK"
```

Replace the `evaluate_trade` method signature and body:
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
    if self._drawdown_triggered:
        return RiskDecision(allowed=False, reason=BlockReason.MAX_DRAWDOWN)
    if self._daily_loss_triggered:
        return RiskDecision(allowed=False, reason=BlockReason.DAILY_LOSS_LIMIT)
    if open_positions_count >= max_open_positions:
        return RiskDecision(allowed=False, reason=BlockReason.MAX_POSITIONS_REACHED)
    if signal != 1:
        return RiskDecision(allowed=False, reason=BlockReason.LOW_CONFIDENCE)
    if probability < confidence_threshold:
        return RiskDecision(allowed=False, reason=BlockReason.LOW_CONFIDENCE)
    if has_open_position:
        return RiskDecision(allowed=False, reason=BlockReason.POSITION_OPEN)
    if not trend_ok:
        return RiskDecision(allowed=False, reason=BlockReason.TREND_FILTER)

    slots_free = max(1, max_open_positions - open_positions_count)
    position_usd = available_capital / slots_free
    if position_usd < 10:
        return RiskDecision(allowed=False, reason=BlockReason.INSUFFICIENT_CAPITAL)

    sl_price, tp_price = self._compute_sl_tp(current_price, atr_value)

    logger.info(
        f"Operación APROBADA | size=${position_usd:.2f} | "
        f"entrada={current_price:.4f} | SL={sl_price:.4f} | TP={tp_price:.4f} "
        f"({'ATR' if atr_value > 0 else 'fijo'})"
    )
    return RiskDecision(
        allowed=True,
        reason=BlockReason.OK,
        position_size_usd=round(position_usd, 2),
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_risk_manager.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```
pytest tests/ -v
```
Expected: all previous tests still pass.

- [ ] **Step 6: Commit**

```bash
git add risk/manager.py tests/test_risk_manager.py
git commit -m "feat: multi-position risk manager — MAX_POSITIONS_REACHED + slot-based allocation"
```

---

## Task 5: Scheduler — Multi-Symbol Trading Cycle

**Files:**
- Modify: `scheduler/main.py`

**Interfaces:**
- Consumes: `TRADING["symbols"]` (list) from Task 1
- Consumes: `is_model_trained(symbol)`, `predict(df, symbol, threshold)` from Task 2
- Consumes: `get_open_trades_count()` from Task 3
- Consumes: `evaluate_trade(..., open_positions_count, max_open_positions)` from Task 4
- Produces: multi-symbol trading loop, retrain loop, sltp loop

No unit tests for this task — the scheduler integrates live data fetching. Verify via log output after deploy (see Step 6).

- [ ] **Step 1: Update imports in `scheduler/main.py`**

Replace the import block at the top (lines 19–32) with:
```python
from config.settings import EXCHANGE, MODEL, RISK, TELEGRAM, TRADING, settings
from data.fetcher import fetch_historical_data, fetch_latest_candle, fetch_higher_tf_bias
from data.features import calculate_features, calculate_features_inference
from db_layer.models import PortfolioSnapshot, init_db
from db_layer.repository import (
    get_latest_snapshot,
    get_open_trade,
    get_open_trades_count,
    get_today_pnl,
    save_snapshot,
)
from execution.broker import Broker
from models.predictor import is_model_trained, predict
from models.trainer import train_model
from notifications.telegram_bot import TelegramNotifier
from risk.manager import RiskManager
```

- [ ] **Step 2: Update `_initialize()` to read `symbols` list**

Replace the log line near the end of `_initialize()` (around line 106):
```python
logger.info(
    f"Modo: {TRADING['mode'].upper()} | Símbolos: {TRADING['symbols']} | "
    f"Capital: ${current_capital:,.2f} | Timeframe: {TRADING['timeframe']}"
)
```

- [ ] **Step 3: Update `retrain_job()` to loop over all symbols**

Replace the body of `retrain_job()` from line 113 onwards with:
```python
def retrain_job() -> None:
    """Reentrenamiento nocturno (02:30 AM Lima). Separado del ciclo de trading."""
    global _last_retrain_date
    logger.info("=" * 50)
    logger.info("REENTRENAMIENTO NOCTURNO INICIADO")

    days_since = (date.today() - _last_retrain_date).days
    retrain_every = MODEL.get("retrain_days", 14)
    force = os.environ.get("FORCE_RETRAIN", "").lower() in ("true", "1", "yes")

    if not force and days_since < retrain_every:
        logger.info(f"Reentrenamiento no necesario (días desde último: {days_since}/{retrain_every})")
        return
    if force:
        logger.info("FORCE_RETRAIN=true — reentrenamiento forzado")

    symbols = TRADING.get("symbols", [TRADING.get("symbol", "BTC/USDT")])

    for symbol in symbols:
        logger.info(f"Entrenando modelo para {symbol}...")
        try:
            df = fetch_historical_data(
                symbol=symbol,
                days=MODEL.get("historical_days", 60),
                interval=TRADING["timeframe"],
            )
            df_features = calculate_features(df, MODEL.get("target_threshold", 0.002))
            train_model(df_features, symbol=symbol)
            logger.info(f"Modelo {symbol} entrenado correctamente.")
        except Exception as exc:
            logger.error(f"Error entrenando {symbol}: {exc}", exc_info=True)
            notifier.notify_error(f"Error reentrenamiento {symbol}: {exc}")

    _last_retrain_date = date.today()

    from db_layer.models import ModelMetric, engine
    from sqlalchemy.orm import Session
    with Session(engine) as s:
        last_metric = s.query(ModelMetric).order_by(ModelMetric.timestamp.desc()).first()
        if last_metric:
            notifier.notify_retrain(last_metric.accuracy, last_metric.f1_score)

    logger.info("Reentrenamiento nocturno completado.")
```

- [ ] **Step 4: Update `sltp_check_cycle()` to loop over all symbols**

Replace the entire `sltp_check_cycle()` function:
```python
def sltp_check_cycle() -> None:
    """Verifica SL/TP y trailing stop cada 5 minutos para todos los símbolos."""
    if broker is None:
        return
    symbols = TRADING.get("symbols", [TRADING.get("symbol", "BTC/USDT")])
    for symbol in symbols:
        try:
            if not get_open_trade(symbol):
                continue
            exit_reason = broker.check_and_manage_open_position(symbol, risk_manager)
            if exit_reason:
                from db_layer.repository import get_all_trades
                trades = get_all_trades()
                if trades:
                    last_trade = trades[0]
                    notifier.notify_trade_closed(
                        symbol=symbol,
                        entry=last_trade.price,
                        exit_price=last_trade.close_price or 0,
                        pnl=last_trade.pnl or 0,
                        pnl_pct=last_trade.pnl_pct or 0,
                        reason=exit_reason,
                        mode=TRADING["mode"],
                    )
                logger.info(f"[{symbol}] SL/TP/Trailing ejecutado: motivo={exit_reason}")
        except Exception as exc:
            logger.error(f"[{symbol}] Error en verificación SL/TP: {exc}", exc_info=True)
```

- [ ] **Step 5: Replace `trading_cycle()` with multi-symbol loop**

Replace the entire `trading_cycle()` function:
```python
def trading_cycle() -> None:
    """
    Ciclo principal cada 15 minutos.
    Itera sobre todos los símbolos configurados y evalúa señales de entrada/salida.
    """
    logger.info(f"{'─'*50}")
    logger.info(f"CICLO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'─'*50}")

    symbols = TRADING.get("symbols", [TRADING.get("symbol", "BTC/USDT")])
    initial_capital = TRADING["capital"]
    max_open_pos = RISK.get("max_open_positions", 1)
    confidence_threshold = MODEL.get("prediction_threshold", 0.65)

    snapshot = get_latest_snapshot()
    cash = snapshot.cash_usd if snapshot else initial_capital
    total_position_value = 0.0

    # DeepSeek context — once per cycle, applies to all symbols
    from analysis.deepseek import get_market_context
    market_ctx = get_market_context()
    adjusted_threshold = round(confidence_threshold + market_ctx.modifier, 4)
    adjusted_threshold = max(0.55, min(0.90, adjusted_threshold))
    logger.info(
        f"Umbral ajustado: {adjusted_threshold:.0%} "
        f"(base={confidence_threshold:.0%}, mod={market_ctx.modifier:+.0%}, "
        f"mercado={market_ctx.sentiment.upper()})"
    )

    today_pnl = get_today_pnl()
    _reset_daily_state(cash + total_position_value)
    risk_manager.update_portfolio_state(
        current_value=cash + total_position_value,
        daily_pnl=today_pnl,
        initial_daily_value=_initial_daily_value,
    )

    for symbol in symbols:
        try:
            logger.info(f"── {symbol} ──")

            if not is_model_trained(symbol):
                logger.warning(f"[{symbol}] Modelo no entrenado. Saltando.")
                continue

            current_price = broker.get_current_price(symbol)
            if not current_price:
                logger.warning(f"[{symbol}] Precio no disponible. Saltando.")
                continue

            open_trade = get_open_trade(symbol)
            if open_trade:
                total_position_value += open_trade.quantity * current_price

            # Verificar SL/TP y trailing de posición abierta en este símbolo
            if open_trade:
                exit_reason = broker.check_and_manage_open_position(symbol, risk_manager)
                if exit_reason:
                    closed_check = get_open_trade(symbol)
                    if not closed_check:
                        from db_layer.repository import get_all_trades
                        last_trade = get_all_trades()[0]
                        notifier.notify_trade_closed(
                            symbol=symbol,
                            entry=last_trade.price,
                            exit_price=last_trade.close_price or current_price,
                            pnl=last_trade.pnl or 0,
                            pnl_pct=last_trade.pnl_pct or 0,
                            reason=exit_reason,
                            mode=TRADING["mode"],
                        )
                        cash += (last_trade.close_price or current_price) * (last_trade.quantity or 0)
                        open_trade = None

            # Datos históricos para predicción
            df_pred = fetch_historical_data(symbol, days=30, interval=TRADING["timeframe"])
            try:
                df_live_candle = fetch_latest_candle(symbol, EXCHANGE)
                if df_live_candle is not None and not df_live_candle.empty:
                    import pandas as pd
                    df_pred = pd.concat([df_pred, df_live_candle])
                    df_pred = df_pred[~df_pred.index.duplicated(keep="last")]
            except Exception:
                pass

            # ATR dinámico
            atr_value = 0.0
            try:
                df_feat = calculate_features_inference(df_pred)
                if not df_feat.empty:
                    atr_pct = float(df_feat["atr_pct"].iloc[-1])
                    atr_value = atr_pct * current_price
                    logger.info(f"[{symbol}] ATR: {atr_pct:.4f} ({atr_value:.2f} USD)")
            except Exception as exc:
                logger.warning(f"[{symbol}] No se pudo calcular ATR: {exc}")

            # Filtro de tendencia 4h
            trend_ok = fetch_higher_tf_bias(symbol)
            if not trend_ok:
                logger.info(f"[{symbol}] Filtro 4h: BAJISTA — no se abrirán posiciones")

            prediction = predict(df_pred, symbol=symbol, threshold=adjusted_threshold)
            if not prediction:
                logger.warning(f"[{symbol}] Predicción no disponible. Saltando.")
                continue

            logger.info(
                f"[{symbol}] Predicción: {'COMPRAR' if prediction['signal'] == 1 else 'VENDER/ESPERAR'} "
                f"| confianza={prediction['probability']:.1%} "
                f"| 4h={'ALCISTA' if trend_ok else 'BAJISTA'}"
            )

            # Cerrar posición si el modelo predice venta
            if open_trade and prediction["signal"] == 0:
                closed = broker.close_position(
                    symbol=symbol,
                    trade_id=open_trade.id,
                    quantity=open_trade.quantity,
                    current_price=current_price,
                    reason="signal",
                )
                if closed:
                    notifier.notify_trade_closed(
                        symbol=symbol,
                        entry=open_trade.price,
                        exit_price=current_price,
                        pnl=closed.pnl or 0,
                        pnl_pct=closed.pnl_pct or 0,
                        reason="signal",
                        mode=TRADING["mode"],
                    )
                    cash += current_price * open_trade.quantity
                    open_trade = None
                    logger.info(f"[{symbol}] Posición cerrada por señal de venta.")

            # Evaluar apertura de nueva posición
            if not open_trade:
                open_positions_count = get_open_trades_count()
                decision = risk_manager.evaluate_trade(
                    signal=prediction["signal"],
                    probability=prediction["probability"],
                    confidence_threshold=adjusted_threshold,
                    current_price=current_price,
                    available_capital=cash,
                    has_open_position=False,
                    atr_value=atr_value,
                    trend_ok=trend_ok,
                    open_positions_count=open_positions_count,
                    max_open_positions=max_open_pos,
                )

                if decision.allowed:
                    trade = broker.open_position(symbol, decision, current_price)
                    if trade:
                        cash -= decision.position_size_usd
                        notifier.notify_trade_opened(
                            symbol=symbol,
                            price=current_price,
                            quantity=trade.quantity,
                            sl=decision.stop_loss_price,
                            tp=decision.take_profit_price,
                            mode=TRADING["mode"],
                            market_sentiment=market_ctx.sentiment,
                        )
                else:
                    logger.info(f"[{symbol}] Bloqueado: {decision.reason.value}")
                    if decision.reason.name in ("DAILY_LOSS_LIMIT", "MAX_DRAWDOWN"):
                        notifier.notify_circuit_breaker(
                            reason=decision.reason.value,
                            value=abs(today_pnl / initial_capital * 100),
                        )

        except Exception as exc:
            logger.error(f"[{symbol}] Error inesperado: {exc}", exc_info=True)
            notifier.notify_error(f"[{symbol}] {exc}")

    # Snapshot global al final del ciclo
    open_trades_all = [get_open_trade(s) for s in symbols]
    position_value_total = sum(
        t.quantity * (broker.get_current_price(t.symbol) or t.price)
        for t in open_trades_all if t
    )
    _save_portfolio_snapshot(cash, position_value_total, initial_capital)
    logger.info(f"Snapshot guardado | total=${cash + position_value_total:,.2f}")
```

- [ ] **Step 6: Verify smoke test — run one cycle locally**

```bash
cd "C:\Users\onunez\OneDrive - MT INDUSTRIAL S.A.C\Escritorio\ProyectosNoCodeCreator\AsistenteTrading"
python -c "
import sys; sys.path.insert(0, '.')
from scheduler.main import _initialize, trading_cycle
_initialize()
trading_cycle()
"
```
Expected: logs show `── BTC/USDT ──`, `── ETH/USDT ──`, etc. for each symbol. No `KeyError: 'symbol'`. Symbols without trained models show `Modelo no entrenado. Saltando.`

- [ ] **Step 7: Commit**

```bash
git add scheduler/main.py
git commit -m "feat: multi-symbol trading cycle — loops over all configured symbols"
```

---

## Task 6: Dashboard — Asset Grid + Multi-Position Table

**Files:**
- Modify: `dashboard/app.py`

**Interfaces:**
- Consumes: `TRADING["symbols"]` (list) from Task 1
- Consumes: `is_model_trained(symbol)` from Task 2
- Consumes: `get_model_metric_by_symbol(symbol)` from Task 3
- Consumes: `open_trades` list (already multi-symbol via `get_all_trades()`)

No unit tests — visual output verified by running Streamlit locally.

- [ ] **Step 1: Update config reads at the top of `dashboard/app.py`**

Find this block (around line 652–654):
```python
initial_capital = float(TRADING.get("capital", 1000))
mode   = TRADING.get("mode", "paper")
symbol = TRADING.get("symbol", "BTC/USDT")
```
Replace with:
```python
initial_capital = float(TRADING.get("capital", 1000))
mode    = TRADING.get("mode", "paper")
symbols = TRADING.get("symbols", [TRADING.get("symbol", "BTC/USDT")])
symbol  = symbols[0]  # primary symbol for BTC chart header
```

- [ ] **Step 2: Update model status in header (line ~676)**

Find:
```python
model_ok  = is_model_trained()
```
Replace with:
```python
trained_count = sum(1 for s in symbols if is_model_trained(s))
model_ok = trained_count > 0
model_text = f"IA: {trained_count}/{len(symbols)}" if model_ok else "Sin modelos"
```

- [ ] **Step 3: Add `load_asset_data()` cached function**

After the `load_model_metrics()` function (around line 471), add:

```python
@st.cache_data(ttl=60)
def load_asset_data(syms: tuple):
    """Fetches current price and 24h change for each symbol via yfinance."""
    from data.fetcher import _SYMBOL_MAP
    results = {}
    for sym in syms:
        ticker = _SYMBOL_MAP.get(sym, sym.replace("/", "-").replace("USDT", "USD"))
        try:
            df = yf.download(ticker, period="2d", interval="1h",
                             progress=False, auto_adjust=True)
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)
            if df is not None and not df.empty:
                close = df["Close"].squeeze()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                cur = float(close.iloc[-1])
                prev = float(close.iloc[-25]) if len(close) >= 25 else float(close.iloc[0])
                pct = (cur / prev - 1) * 100 if prev else 0.0
                results[sym] = {"price": cur, "pct": pct}
            else:
                results[sym] = {"price": 0.0, "pct": 0.0}
        except Exception:
            results[sym] = {"price": 0.0, "pct": 0.0}
    return results
```

- [ ] **Step 4: Add `load_asset_signals()` cached function**

After `load_asset_data()`, add:

```python
@st.cache_data(ttl=300)
def load_asset_signals(syms: tuple):
    """Returns per-symbol ML signal info from latest ModelMetric in DB."""
    from db_layer.repository import get_model_metric_by_symbol
    results = {}
    for sym in syms:
        trained = is_model_trained(sym)
        metric = get_model_metric_by_symbol(sym) if trained else None
        results[sym] = {
            "trained": trained,
            "accuracy": metric.accuracy if metric else None,
            "f1": metric.f1_score if metric else None,
        }
    return results
```

- [ ] **Step 5: Load new data in the data loading block**

Find the data loading block (around lines 646–650):
```python
df_mkt, price, ch, ch_pct, high_24h, low_24h, vol_24h = get_market_data()
snapshot, trades, snapshots, today_pnl = load_db_data()
fear_greed_val, fear_greed_label = load_fear_greed()
mkt_ctx = load_market_context()
model_metrics = load_model_metrics()
```
Add after these lines:
```python
asset_prices  = load_asset_data(tuple(symbols))
asset_signals = load_asset_signals(tuple(symbols))
```

- [ ] **Step 6: Add CSS for asset grid cards**

Inside the `<style>` block (after the last existing CSS rule, before the closing `</style>`), add:
```css
  /* ── Asset grid cards ── */
  .asset-card {
    background: #181A20;
    border: 1px solid #2B2F36;
    border-radius: 4px;
    padding: 10px 12px;
    margin-bottom: 6px;
    min-height: 110px;
  }
  .asset-card-sym {
    font-size: 0.78rem; font-weight: 700;
    color: #EAECEF; margin-bottom: 2px;
  }
  .asset-card-price {
    font-size: 1.05rem; font-weight: 700;
    color: #EAECEF; font-variant-numeric: tabular-nums;
    margin-bottom: 2px;
  }
  .asset-card-badge {
    font-size: 0.62rem; font-weight: 700;
    padding: 1px 6px; border-radius: 2px;
    display: inline-block; margin-bottom: 4px;
  }
  .badge-crypto { background: #2B2F36; color: #F0B90B; }
  .badge-forex  { background: #1E3A5F; color: #56CCF2; }
  .badge-gold   { background: #3A2E1E; color: #F2C94C; }
  .badge-oil    { background: #1E2E1E; color: #6FCF97; }
```

- [ ] **Step 7: Add asset grid section to the dashboard**

After the main header divider (around line 753, after the `st.markdown` with `border-bottom`), insert the asset grid before the `col_chart, col_panel = st.columns([70, 30])` line:

```python
# ── Asset Grid ──────────────────────────────────────────────────────────────────
ASSET_CLASS = {
    "BTC/USDT": ("Crypto", "badge-crypto"),
    "ETH/USDT": ("Crypto", "badge-crypto"),
    "SOL/USDT": ("Crypto", "badge-crypto"),
    "BNB/USDT": ("Crypto", "badge-crypto"),
    "EUR/USD":  ("Forex",  "badge-forex"),
    "GBP/USD":  ("Forex",  "badge-forex"),
    "XAU/USD":  ("Gold",   "badge-gold"),
    "XTI/USD":  ("Oil",    "badge-oil"),
}

st.markdown(
    "<div style='font-size:0.7rem;font-weight:700;color:#848E9C;"
    "text-transform:uppercase;letter-spacing:1.5px;padding:6px 0 4px 0'>"
    "MERCADOS</div>",
    unsafe_allow_html=True,
)

grid_cols = st.columns(len(symbols))
for idx, sym in enumerate(symbols):
    with grid_cols[idx]:
        info   = asset_prices.get(sym, {"price": 0.0, "pct": 0.0})
        sig    = asset_signals.get(sym, {"trained": False, "accuracy": None})
        cls, badge = ASSET_CLASS.get(sym, ("Asset", "badge-crypto"))
        pct    = info["pct"]
        prc    = info["price"]
        pct_color = "#0ECB81" if pct >= 0 else "#F6465D"
        pct_arrow = "▲" if pct >= 0 else "▼"
        if sig["trained"] and sig["accuracy"] is not None:
            sig_text = f"IA: {sig['accuracy']:.0%} acc"
            sig_color = "#0ECB81" if sig["accuracy"] > 0.55 else "#848E9C"
        else:
            sig_text = "SIN MODELO"
            sig_color = "#474D57"

        # Format price: large numbers with commas, small numbers with 4 decimals
        if prc >= 1:
            prc_str = f"${prc:,.2f}"
        elif prc > 0:
            prc_str = f"${prc:.4f}"
        else:
            prc_str = "—"

        st.markdown(
            f'<div class="asset-card">'
            f'<div class="asset-card-sym">{sym}</div>'
            f'<span class="asset-card-badge {badge}">{cls}</span>'
            f'<div class="asset-card-price">{prc_str}</div>'
            f'<div style="font-size:0.75rem;color:{pct_color};font-weight:600">'
            f'{pct_arrow} {abs(pct):.2f}%</div>'
            f'<div style="font-size:0.68rem;color:{sig_color};margin-top:3px">'
            f'{sig_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='border-bottom:1px solid #2B2F36;margin:6px 0 8px 0'></div>",
            unsafe_allow_html=True)
```

- [ ] **Step 8: Update open positions display in the panel**

Find in `col_panel` the positions state logic (around line 782):
```python
if open_trades:
    btn_cls  = "bn-btn-buy"
    btn_txt  = "▶ EN POSICIÓN — COMPRANDO"
    state_detail = "Posición BTC abierta"
```
Replace `state_detail`:
```python
if open_trades:
    btn_cls  = "bn-btn-buy"
    btn_txt  = f"▶ {len(open_trades)} POSICIÓN(ES) ACTIVA(S)"
    state_detail = " | ".join(t.symbol for t in open_trades)
```

Find the `_row("Operaciones", ...)` line (around line 843):
```python
+ _row("Operaciones",  f"{n_ops} cerradas · {len(open_trades)} abierta", border=False)
```
Replace with:
```python
+ _row("Operaciones",  f"{n_ops} cerradas · {len(open_trades)} abiertas", border=False)
```

After the rows HTML `st.markdown` block (around line 848), add the open positions table:
```python
if open_trades:
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;color:#848E9C;"
        "text-transform:uppercase;letter-spacing:1.5px;padding:8px 0 4px 0'>"
        "POSICIONES ABIERTAS</div>",
        unsafe_allow_html=True,
    )
    rows_pos = ""
    for ot in open_trades:
        cur_px = asset_prices.get(ot.symbol, {}).get("price", ot.price) or ot.price
        pnl_pct = (cur_px / ot.price - 1) * 100 if ot.price else 0.0
        pnl_color = "#0ECB81" if pnl_pct >= 0 else "#F6465D"
        sl_str = f"${ot.stop_loss:,.2f}" if ot.stop_loss else "—"
        tp_str = f"${ot.take_profit:,.2f}" if ot.take_profit else "—"
        trail  = "✓" if ot.trailing_active else "—"
        rows_pos += (
            f"<tr>"
            f"<td>{ot.symbol}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>${ot.price:,.2f}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>${cur_px:,.2f}</td>"
            f"<td style='color:{pnl_color};font-weight:600'>{pnl_pct:+.2f}%</td>"
            f"<td style='color:#848E9C'>{sl_str}</td>"
            f"<td style='color:#848E9C'>{tp_str}</td>"
            f"<td style='color:#F0B90B;text-align:center'>{trail}</td>"
            f"</tr>"
        )
    st.markdown(
        f'<div style="background:#181A20;border:1px solid #2B2F36;border-radius:4px;overflow:hidden">'
        f'<table class="bn-table" style="width:100%">'
        f'<thead><tr>'
        f'<th>Par</th><th>Entrada</th><th>Actual</th>'
        f'<th>PnL%</th><th>SL</th><th>TP</th><th>Trail</th>'
        f'</tr></thead>'
        f'<tbody>{rows_pos}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 9: Verify dashboard renders correctly**

```bash
streamlit run dashboard/app.py
```
Open browser. Verify:
- Asset grid shows 8 cards with prices and class badges
- Prices update on "Actualizar"
- Open positions table appears when there are open trades
- No `KeyError: 'symbol'` errors in terminal

- [ ] **Step 10: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard — 8-asset signal grid + multi-position table"
```

---

## Task 7: Migration & First-Run Checklist

**Files:** No code changes — deploy steps only.

- [ ] **Step 1: Run full test suite one last time**

```
pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 2: Set `FORCE_RETRAIN=true` in EasyPanel environment variables**

In EasyPanel → App → Environment Variables, add:
```
FORCE_RETRAIN=true
```
This triggers training of all 8 models on first startup. Remove after the first successful deploy.

- [ ] **Step 3: Delete old model file (optional)**

The old `models/saved/model.pkl` (BTC-only, unnamed) is no longer used. It can be deleted safely:
```bash
del "models\saved\model.pkl"
```
Or leave it — it won't interfere.

- [ ] **Step 4: Deploy to EasyPanel and verify logs**

After deploy, watch logs for:
```
── BTC/USDT ──
── ETH/USDT ──
── SOL/USDT ──
── BNB/USDT ──
── EUR/USD ──
── GBP/USD ──
── XAU/USD ──
── XTI/USD ──
Snapshot guardado | total=$1,000.00
```
And during `FORCE_RETRAIN` startup:
```
Entrenando modelo para BTC/USDT...
Modelo BTC/USDT entrenado correctamente.
Entrenando modelo para ETH/USDT...
...
```

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "feat: multi-asset trading — 8 symbols, 3 max positions, per-symbol ML models"
```
