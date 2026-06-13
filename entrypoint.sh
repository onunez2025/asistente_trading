#!/bin/bash
set -e

# Inicia el bot en segundo plano
echo "[entrypoint] Iniciando bot de trading..."
python scheduler/main.py &

# Inicia el dashboard Streamlit en primer plano (mantiene el contenedor vivo)
echo "[entrypoint] Iniciando dashboard Streamlit en puerto 8501..."
exec python -m streamlit run dashboard/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
