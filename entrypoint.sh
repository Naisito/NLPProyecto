#!/bin/bash
set -e

# Solo el servicio API necesita los modelos HuggingFace.
# El frontend (Streamlit) no los usa directamente.
if [ "${RUN_PREFETCH:-0}" = "1" ]; then
    python /app/prefetch_models.py
    # Tras la descarga, activar modo offline para evitar llamadas a HuggingFace.
    export HF_HUB_OFFLINE=1
fi

# Ejecutar el comando pasado como argumento (uvicorn o streamlit).
exec "$@"
