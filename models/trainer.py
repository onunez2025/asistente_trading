import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier

from data.features import FEATURE_COLUMNS
from database.models import ModelMetric, init_db
from database.repository import save_model_metric

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_PATH = Path(__file__).parent / "saved" / "model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5      # Folds de Walk-Forward Validation
N_TRIALS = 30     # Iteraciones de optimización de hiperparámetros


def _walk_forward_score(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Evalúa el modelo con Walk-Forward Validation (respeta el orden temporal)."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    f1_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, zero_division=0))
    return float(np.mean(f1_scores))


def _objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Función objetivo para Optuna: busca los mejores hiperparámetros."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = LGBMClassifier(**params)
    return _walk_forward_score(model, X, y)


def train_model(df: pd.DataFrame, symbol: str = "BTC/USDT") -> LGBMClassifier:
    """
    Entrena el modelo de IA con Walk-Forward Validation y optimización de hiperparámetros.
    Guarda el modelo en disco y registra las métricas en la base de datos.
    """
    init_db()

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available) < len(FEATURE_COLUMNS):
        missing = set(FEATURE_COLUMNS) - set(available)
        logger.warning(f"Features faltantes, se omitirán: {missing}")

    X = df[available].values
    y = df["target"].values

    logger.info(f"Iniciando entrenamiento | filas={len(X)} | features={len(available)}")
    logger.info(f"Distribución del target: compra={y.sum()} | venta={len(y)-y.sum()}")

    # Baseline: modelo que siempre predice la clase más frecuente
    dummy = DummyClassifier(strategy="most_frequent")
    baseline_f1 = _walk_forward_score(dummy, X, y)
    dummy.fit(X, y)
    baseline_acc = accuracy_score(y, dummy.predict(X))
    logger.info(f"Baseline accuracy={baseline_acc:.3f} | f1={baseline_f1:.3f}")

    # Optimización de hiperparámetros con Optuna
    logger.info(f"Optimizando hiperparámetros ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: _objective(trial, X, y), n_trials=N_TRIALS)

    best_params = study.best_params
    best_params.update({"random_state": 42, "verbose": -1, "n_jobs": -1})
    logger.info(f"Mejores parámetros: {best_params}")

    # Entrenamiento final con todos los datos y los mejores parámetros
    final_model = LGBMClassifier(**best_params)
    final_model.fit(X, y)

    # Métricas finales (Walk-Forward sobre todo el conjunto)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    all_preds, all_true = [], []
    for train_idx, val_idx in tscv.split(X):
        final_model.fit(X[train_idx], y[train_idx])
        all_preds.extend(final_model.predict(X[val_idx]))
        all_true.extend(y[val_idx])

    acc = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, zero_division=0)
    rec = recall_score(all_true, all_preds, zero_division=0)
    f1 = f1_score(all_true, all_preds, zero_division=0)

    logger.info("=" * 50)
    logger.info("RESULTADOS DEL MODELO DE IA")
    logger.info(f"  Accuracy:  {acc:.3f}  (baseline: {baseline_acc:.3f})")
    logger.info(f"  Precision: {prec:.3f}")
    logger.info(f"  Recall:    {rec:.3f}")
    logger.info(f"  F1-Score:  {f1:.3f}  (baseline: {baseline_f1:.3f})")
    logger.info("=" * 50)

    if f1 <= baseline_f1:
        logger.warning(
            "El modelo NO supera al baseline. Las señales pueden no ser confiables. "
            "Considera más datos o ajustar el target_threshold en settings.yaml."
        )

    # Entrenamiento final con TODOS los datos (para producción)
    final_model.fit(X, y)

    # Guardar modelo
    joblib.dump({"model": final_model, "features": available}, MODEL_PATH)
    logger.info(f"Modelo guardado en {MODEL_PATH}")

    # Guardar métricas en base de datos
    metric = ModelMetric(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        baseline_accuracy=baseline_acc,
        training_rows=len(X),
        symbol=symbol,
    )
    save_model_metric(metric)

    return final_model
