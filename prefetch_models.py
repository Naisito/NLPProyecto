"""
Script de descarga de modelos HuggingFace.

- Se ejecuta al arrancar el contenedor Docker (vía entrypoint.sh).
- Si los modelos ya están en HF_HOME, no descarga nada.
- Si no están (primera vez o caché borrada), los descarga.
- Escribe un fichero .models_ready como marca de descarga completa.
"""

import json
import os
import sys

HF_HOME     = os.environ.get("HF_HOME", "/app/models_cache")
MARKER_FILE = os.path.join(HF_HOME, ".models_ready")

# Leer nombres de modelos desde config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    EMBED_MODEL  = cfg["embeddings"]["model_name"]
    RERANK_MODEL = cfg["reranker"]["model_name"]
    RERANK_ON    = cfg["reranker"].get("enabled", True)
except Exception as e:
    print(f"[prefetch] No se pudo leer config.json: {e}. Usando valores por defecto.")
    EMBED_MODEL  = "BAAI/bge-m3"
    RERANK_MODEL = "cross-encoder/ms-marco-multilingual-MiniLM-L12-v2"
    RERANK_ON    = True


def models_ready() -> bool:
    """Devuelve True si el marcador existe Y los ficheros del modelo de embeddings están."""
    if not os.path.exists(MARKER_FILE):
        return False
    # Comprobación extra: que el directorio del modelo de embeddings exista
    hub_dir = os.path.join(HF_HOME, "hub")
    model_slug = "models--" + EMBED_MODEL.replace("/", "--")
    return os.path.isdir(os.path.join(hub_dir, model_slug))


def download():
    print(f"[prefetch] Descargando modelos en {HF_HOME} ...")

    # --- Modelo de embeddings ---
    print(f"[prefetch]   → {EMBED_MODEL}")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(EMBED_MODEL, cache_folder=HF_HOME)
        print(f"[prefetch]   ✓ {EMBED_MODEL} listo.")
    except Exception as e:
        print(f"[prefetch]   ✗ Error descargando embeddings: {e}")
        sys.exit(1)   # Crítico: sin embeddings el sistema no arranca

    # --- Modelo de reranking (opcional) ---
    if RERANK_ON:
        print(f"[prefetch]   → {RERANK_MODEL}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=RERANK_MODEL,
                cache_dir=os.path.join(HF_HOME, "hub"),
            )
            print(f"[prefetch]   ✓ {RERANK_MODEL} listo.")
        except Exception as e:
            print(f"[prefetch]   ⚠ Reranker no disponible (continuando sin él): {e}")
            # No es crítico: el sistema funciona sin reranker
    else:
        print(f"[prefetch]   → Reranker desactivado en config.json, saltando.")

    # Escribir marcador de éxito
    os.makedirs(HF_HOME, exist_ok=True)
    with open(MARKER_FILE, "w") as f:
        f.write("ok\n")
    print("[prefetch] ✅ Modelos listos. Caché guardada en volumen.")


if __name__ == "__main__":
    if models_ready():
        print("[prefetch] ✅ Modelos ya en caché. Saltando descarga.")
    else:
        download()
