# Fix 5 Critical Bugs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corregir los 5 bugs críticos que hacen que el bot opere con datos retrasados, nunca cierre posiciones por señal, y pierda los circuit breakers en cada reinicio.

**Architecture:** Cambios quirúrgicos en 4 archivos existentes (`repository.py`, `features.py`, `predictor.py`, `risk/manager.py`, `scheduler/main.py`). Sin nuevas abstracciones. Sin tests formales (el proyecto no tiene suite de tests); cada tarea incluye verificación manual por log.

**Tech Stack:** Python 3.11, SQLAlchemy 2.x, LightGBM, APScheduler, pandas, SQLite

## Global Constraints

- No crear archivos nuevos salvo lo indicado
- No refactorizar código fuera del alcance del bug
- Cada tarea termina con un commit
- Todos los cambios deben ser compatibles con el despliegue Docker/EasyPanel existente
- No cambiar la firma pública de funciones usadas en backtesting (`calculate_features` sigue igual)

---

### Task 1: Fix SQLAlchemy — objetos detached

**Files:**
- Modify: `database/repository.py:14-15`

**Interfaces:**
- Produces: sesiones con `expire_on_commit=False` — todos los objetos ORM retornados conservan sus atributos después de que la sesión se cierra

**Problema:** `_session()` crea una `Session` con `expire_on_commit=True` (default). Tras un `commit()` dentro de `with _session() as s:`, todos los atributos del objeto quedan marcados como "expirados". El `s.refresh(trade)` los recarga, pero al salir del `with`, la sesión se cierra y el objeto queda detached. Si SQLAlchemy intenta re-cargar un atributo expirado sobre un objeto detached, lanza `DetachedInstanceError`.

- [ ] **Step 1: Modificar `_session()` en `database/repository.py`**

Cambiar líneas 14-15 de:
```python
def _session() -> Session:
    return Session(engine)
```
a:
```python
def _session() -> Session:
    return Session(engine, expire_on_commit=False)
```

- [ ] **Step 2: Verificar que no hay errores de import**

```bash
python -c "from db_layer.repository import close_trade, get_open_trade; print('OK')"
```
Esperado: `OK`

- [ ] **Step 3: Commit**

```bash
git add database/repository.py
git commit -m "fix: expire_on_commit=False para evitar DetachedInstanceError"
```

---

### Task 2: Fix lag de 1 vela — separar inferencia de entrenamiento

**Files:**
- Modify: `data/features.py` — añadir función `calculate_features_inference()`
- Modify: `models/predictor.py:40` — usar la nueva función

**Interfaces:**
- Consumes: `df_raw: pd.DataFrame` con columnas OHLCV
- Produces: `calculate_features_inference(df) -> pd.DataFrame` — mismo DataFrame con features calculadas, SIN filtrar por target (última fila siempre presente)

**Problema:** `calculate_features()` calcula `target = close.shift(-1) / close - 1`, luego filtra filas donde `|future_return| <= threshold` y hace `dropna(subset=FEATURE_COLS + ["target"])`. Esto siempre elimina la última fila (su `shift(-1)` es NaN). `predict()` llama a esta función y opera sobre la penúltima vela, con 1h de retraso.

- [ ] **Step 1: Añadir `calculate_features_inference` en `data/features.py`**

Insertar después de la función `calculate_features` (después de la línea 92):

```python

def calculate_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Igual que calculate_features pero sin calcular ni filtrar por target.
    Usar SOLO para predicción en vivo — la última fila siempre está presente.
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()
    stoch = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi"] = stoch.stochrsi_k()

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

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    df["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df["bb_width"] = (bb_upper - bb_lower) / (bb.bollinger_mavg() + 1e-10)

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"] = atr / (close + 1e-10)

    vol_sma = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / (vol_sma + 1e-10)

    df["candle_body"] = (close - df["open"]) / (df["open"] + 1e-10)
    df["candle_range"] = (high - low) / (df["open"] + 1e-10)

    df = df.dropna(subset=FEATURE_COLUMNS)
    return df
```

- [ ] **Step 2: Modificar `models/predictor.py` para usar la nueva función**

Cambiar línea 8 de:
```python
from data.features import FEATURE_COLUMNS, calculate_features
```
a:
```python
from data.features import FEATURE_COLUMNS, calculate_features_inference
```

Cambiar líneas 40-41 de:
```python
        df_features = calculate_features(df_raw)
        if df_features.empty:
```
a:
```python
        df_features = calculate_features_inference(df_raw)
        if df_features.empty:
```

- [ ] **Step 3: Verificar que predict() devuelve la última fila correcta**

```bash
python -c "
from data.features import calculate_features_inference
import pandas as pd, numpy as np
idx = pd.date_range('2024-01-01', periods=250, freq='1h')
df = pd.DataFrame({'open':100,'high':102,'low':99,'close':101,'volume':1000}, index=idx)
df['close'] = df['close'] + np.random.randn(250) * 0.5
result = calculate_features_inference(df)
print('Ultima fila del df original:', df.index[-1])
print('Ultima fila del resultado:', result.index[-1])
print('Coinciden:', df.index[-1] == result.index[-1])
"
```
Esperado: `Coinciden: True`

- [ ] **Step 4: Commit**

```bash
git add data/features.py models/predictor.py
git commit -m "fix: calculate_features_inference sin shift(-1) — predice sobre la vela actual"
```

---

### Task 3: Persistir peak_value del circuit breaker entre reinicios

**Files:**
- Modify: `risk/manager.py:33-43` — aceptar `peak_value` inicial como parámetro
- Modify: `scheduler/main.py:66-69` — pasar `peak_value` desde el último snapshot

**Interfaces:**
- `RiskManager.__init__(risk_cfg, initial_capital, peak_value=None)` — si `peak_value` es None usa `initial_capital`

**Problema:** `RiskManager.__init__` siempre pone `self.peak_value = initial_capital`. Si el bot se reinicia con el portfolio en drawdown, el nuevo `peak_value` es el valor actual (bajo), no el pico histórico. Los circuit breakers quedan ciegos al historial de pérdidas. El `PortfolioSnapshot` ya guarda `peak_value` en cada snapshot.

- [ ] **Step 1: Modificar `RiskManager.__init__` en `risk/manager.py`**

Cambiar líneas 33-43 de:
```python
    def __init__(self, risk_cfg: dict, initial_capital: float):
        self.max_position_size = risk_cfg["max_position_size"]
        self.stop_loss_pct = risk_cfg["stop_loss"]
        self.take_profit_pct = risk_cfg["take_profit"]
        self.daily_loss_limit = risk_cfg["daily_loss_limit"]
        self.max_drawdown = risk_cfg["max_drawdown"]

        self.initial_capital = initial_capital
        self.peak_value = initial_capital
        self._daily_loss_triggered = False
        self._drawdown_triggered = False
```
a:
```python
    def __init__(self, risk_cfg: dict, initial_capital: float, peak_value: float = None):
        self.max_position_size = risk_cfg["max_position_size"]
        self.stop_loss_pct = risk_cfg["stop_loss"]
        self.take_profit_pct = risk_cfg["take_profit"]
        self.daily_loss_limit = risk_cfg["daily_loss_limit"]
        self.max_drawdown = risk_cfg["max_drawdown"]

        self.initial_capital = initial_capital
        self.peak_value = peak_value if peak_value is not None else initial_capital
        self._daily_loss_triggered = False
        self._drawdown_triggered = False
```

- [ ] **Step 2: Modificar `_initialize()` en `scheduler/main.py` para pasar peak_value**

Cambiar líneas 63-69 de:
```python
    snapshot = get_latest_snapshot()
    current_capital = snapshot.total_value if snapshot else TRADING["capital"]

    risk_manager = RiskManager(
        risk_cfg=RISK,
        initial_capital=current_capital,
    )
```
a:
```python
    snapshot = get_latest_snapshot()
    current_capital = snapshot.total_value if snapshot else TRADING["capital"]
    saved_peak = snapshot.peak_value if snapshot else current_capital

    risk_manager = RiskManager(
        risk_cfg=RISK,
        initial_capital=current_capital,
        peak_value=saved_peak,
    )
    logger.info(f"Peak value restaurado desde BD: ${saved_peak:,.2f}")
```

- [ ] **Step 3: Verificar inicialización**

```bash
python -c "
from risk.manager import RiskManager
rm = RiskManager({'max_position_size':0.5,'stop_loss':0.02,'take_profit':0.04,'daily_loss_limit':0.05,'max_drawdown':0.20}, initial_capital=25.0, peak_value=30.0)
print('peak_value:', rm.peak_value)
assert rm.peak_value == 30.0, 'peak_value no se restauró'
print('OK')
"
```
Esperado: `peak_value: 30.0` y `OK`

- [ ] **Step 4: Commit**

```bash
git add risk/manager.py scheduler/main.py
git commit -m "fix: peak_value del circuit breaker se restaura desde BD al reiniciar"
```

---

### Task 4: Cerrar posición por señal de venta del modelo

**Files:**
- Modify: `scheduler/main.py:244-284` — añadir cierre por señal cuando hay posición abierta

**Interfaces:**
- Consumes: `prediction["signal"]`, `open_trade` (Trade ORM), `broker.close_position()`, `notifier.notify_trade_closed()`
- Precondición: Task 1 (expire_on_commit=False) debe estar aplicado para acceder a `open_trade.price` etc. sin error

**Problema:** El ciclo de trading verifica SL/TP de posición abierta (líneas 209-226) y luego evalúa abrir posición nueva (líneas 249-280). Pero si hay posición abierta y el modelo predice `signal == 0` (VENDER), esa señal es ignorada completamente. Solo SL/TP cierran posiciones. El backtesting sí cierra por señal — diferencia que hace al backtest optimista.

- [ ] **Step 1: Localizar la sección a modificar en `scheduler/main.py`**

El bloque `else:` que empieza en línea ~244 (`logger.info(f"Predicción: ...")`). La lógica nueva va DENTRO del bloque `if prediction:`, justo después de ese `logger.info`, y ANTES del `if not open_trade:`.

- [ ] **Step 2: Modificar `scheduler/main.py` para cerrar por señal**

Dentro del bloque `if prediction:` (línea ~244), reemplazar desde:
```python
            logger.info(
                f"Predicción: {'COMPRAR' if prediction['signal'] == 1 else 'VENDER/ESPERAR'} "
                f"| confianza={prediction['probability']:.1%}"
            )

            # 7. Evaluar si se puede abrir posición nueva
            if not open_trade:
```
a:
```python
            logger.info(
                f"Predicción: {'COMPRAR' if prediction['signal'] == 1 else 'VENDER/ESPERAR'} "
                f"| confianza={prediction['probability']:.1%}"
            )

            # 7. Cerrar posición si el modelo predice VENDER
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
                    position_value = 0.0
                    open_trade = None
                    logger.info("Posición cerrada por señal de venta del modelo.")

            # 8. Evaluar si se puede abrir posición nueva
            if not open_trade:
```

- [ ] **Step 3: Verificar que el archivo parsea sin errores**

```bash
python -c "import scheduler.main; print('OK')"
```
Esperado: `OK` (sin errores de sintaxis)

- [ ] **Step 4: Commit**

```bash
git add scheduler/main.py
git commit -m "fix: cerrar posicion cuando modelo predice VENDER — alinea prod con backtesting"
```

---

### Task 5: Verificar SL/TP cada 5 minutos

**Files:**
- Modify: `scheduler/main.py` — añadir función `sltp_check_cycle()` y registrarla en APScheduler

**Interfaces:**
- Consumes: globales `broker`, `risk_manager`, `notifier`, `TRADING`
- Precondición: `_initialize()` ya ejecutado (globales inicializados)

**Problema:** El bot solo verifica Stop Loss y Take Profit una vez por hora (en el ciclo principal `trading_cycle`). BTC puede moverse ±5% en minutos. Un SL de 2% puede traspasarse y recuperarse sin que el bot lo detecte. La pérdida real puede superar el límite teórico del 2%.

- [ ] **Step 1: Añadir `sltp_check_cycle` en `scheduler/main.py`**

Insertar la nueva función después de `_save_portfolio_snapshot` (antes de `trading_cycle`):

```python
def sltp_check_cycle() -> None:
    """
    Verifica SL/TP de la posición abierta cada 5 minutos.
    No ejecuta predicción ni abre posiciones — solo cierra si corresponde.
    """
    if broker is None:
        return

    symbol = TRADING["symbol"]
    try:
        if not get_open_trade(symbol):
            return

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
            logger.info(f"SL/TP ejecutado fuera del ciclo horario: motivo={exit_reason}")
    except Exception as exc:
        logger.error(f"Error en verificación SL/TP: {exc}", exc_info=True)
```

- [ ] **Step 2: Registrar el job en `start_bot()` en `scheduler/main.py`**

Dentro de `start_bot()`, después de `scheduler.add_job(trading_cycle, ...)`, añadir:

```python
    scheduler.add_job(
        sltp_check_cycle,
        trigger=CronTrigger(minute="*/5"),
        id="sltp_check",
        name="Verificación SL/TP cada 5 minutos",
        misfire_grace_time=60,
        coalesce=True,
    )
    logger.info("Verificación SL/TP activa: cada 5 minutos.")
```

- [ ] **Step 3: Verificar que el scheduler arranca sin errores**

```bash
python -c "
import scheduler.main as m
m._initialize()
print('sltp_check_cycle importable:', callable(m.sltp_check_cycle))
"
```
Esperado: `sltp_check_cycle importable: True`

- [ ] **Step 4: Commit**

```bash
git add scheduler/main.py
git commit -m "feat: verificacion SL/TP cada 5 minutos — reduce ventana de riesgo de 1h a 5min"
```

---

## Deploy final

Una vez todos los commits están en `main`, disparar el webhook de EasyPanel para reconstruir:

```bash
curl -X POST "http://72.61.75.5:3000/api/deploy/6a4675f7df410b5f9905abc0e3bbc34d276879ecf576e25d"
```

Verificar en logs de EasyPanel que el bot arranca con:
- `Peak value restaurado desde BD: $XX.XX`
- `Verificación SL/TP activa: cada 5 minutos.`
- Sin errores `close_trade() got an unexpected keyword argument`
