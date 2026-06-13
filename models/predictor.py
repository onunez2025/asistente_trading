import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from data.features import FEATURE_COLUMNS, calculate_features

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "saved" / "model.pkl"


def is_model_trained() -> bool:
    return MODEL_PATH.exists()


def predict(df_raw: pd.DataFrame, threshold: float = 0.60) -> Optional[dict]:
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
    if not is_model_trained():
        logger.error("El modelo no está entrenado. Ejecuta el entrenamiento primero.")
        return None

    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        features = artifact["features"]

        df_features = calculate_features(df_raw)
        if df_features.empty:
            logger.warning("No hay suficientes datos para calcular features.")
            return None

        last_row = df_features[features].iloc[[-1]]
        proba = model.predict_proba(last_row)[0]
        buy_proba = float(proba[1])   # probabilidad de subida (clase 1)
        signal = 1 if buy_proba >= threshold else 0
        probability = buy_proba

        result = {
            "signal": signal,
            "probability": probability,
            "confidence_ok": buy_proba >= threshold,
        }

        action = "COMPRAR" if signal == 1 else "VENDER/ESPERAR"
        status = "OK" if result["confidence_ok"] else "IGNORADA (baja confianza)"
        logger.info(
            f"Predicción: {action} | confianza={probability:.1%} | {status}"
        )

        return result

    except Exception as exc:
        logger.error(f"Error al generar predicción: {exc}")
        return None
