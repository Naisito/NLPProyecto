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
    RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RERANK_ON    = True


def _model_slug(model_name: str) -> str:
    return "models--" + model_name.replace("/", "--")


def _model_cache_dirs(model_name: str) -> list[str]:
    slug = _model_slug(model_name)
    return [
        os.path.join(HF_HOME, slug),
        os.path.join(HF_HOME, "hub", slug),
    ]


def _snapshot_exists(
    model_name: str,
    required_files: list[str],
    weight_files: list[str] | None = None,
) -> bool:
    """Comprueba si existe al menos un snapshot local utilizable para un modelo."""
    for model_dir in _model_cache_dirs(model_name):
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        for snapshot_name in os.listdir(snapshots_dir):
            snapshot_dir = os.path.join(snapshots_dir, snapshot_name)
            if not os.path.isdir(snapshot_dir):
                continue

            has_required = all(
                os.path.exists(os.path.join(snapshot_dir, filename))
                for filename in required_files
            )
            if not has_required:
                continue

            if weight_files is None:
                return True

            has_any_weight = any(
                os.path.exists(os.path.join(snapshot_dir, filename))
                for filename in weight_files
            )
            if has_any_weight:
                return True

    return False


def models_ready() -> bool:
    """Devuelve True si los modelos necesarios están completos en la caché."""
    if not os.path.exists(MARKER_FILE):
        return False

    embeddings_ready = _snapshot_exists(
        EMBED_MODEL,
        required_files=["config.json"],
        weight_files=["pytorch_model.bin", "model.safetensors"],
    )
    if not embeddings_ready:
        return False

    if not RERANK_ON:
        return True

    return _snapshot_exists(
        RERANK_MODEL,
        required_files=["config.json", "tokenizer.json"],
        weight_files=["pytorch_model.bin", "model.safetensors"],
    )


def download():
    print(f"[prefetch] Descargando modelos en {HF_HOME} ...")

    # --- Modelo de embeddings ---
    print(f"[prefetch]   → {EMBED_MODEL}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=EMBED_MODEL, cache_dir=HF_HOME)
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
                cache_dir=HF_HOME,
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
