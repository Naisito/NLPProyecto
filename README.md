# Generador de Rutas Turisticas — Bilbao / Bizkaia

> **Proyecto NLP — Hito 1** · Grupo VI: Unai de León, Xabier Ballesteros, Oscar Basaguren

Sistema de generación automática de itinerarios turísticos personalizados para **Bilbao y Bizkaia**, implementado como un pipeline **RAG híbrido** que combina recuperación semántica, reranking con cross-encoder, planificación geográfica y generación narrativa en castellano.

El sistema funciona **100% en local**: no requiere claves de API externas ni conexión a internet en tiempo de ejecución.

---

## Índice

1. [Descripción general](#1-descripción-general)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Tecnologías y modelos](#3-tecnologías-y-modelos)
4. [Estructura del proyecto](#4-estructura-del-proyecto)
5. [Instalación](#5-instalación)
6. [Configuración](#6-configuración)
7. [Ejecución desde cero](#7-ejecución)
8. [API REST](#8-api-rest)
9. [Métricas de evaluación](#9-métricas-de-evaluación)
10. [Corpus de datos](#10-corpus-de-datos)
11. [Mejoras respecto a la versión anterior](#11-mejoras-respecto-a-la-versión-anterior)
12. [Referencias](#12-referencias)

---

## 1. Descripción general

El sistema recibe la solicitud del usuario (texto libre o formulario estructurado) y genera automáticamente una ruta turística viable mediante un pipeline de seis etapas completamente automatizado:

```
Texto del usuario
      |
      v
+----------------------------+
| 1. Interpretación (LLM)    |  Ollama (local) extrae preferencias estructuradas
+------------+---------------+
             |
             v
+----------------------------+
| 2. Recuperación (RAG)      |  BAAI/bge-m3 + ChromaDB -> top-k candidatos
+------------+---------------+
             |
             v
+----------------------------+
| 3. Reranking               |  Cross-encoder + preferencias + diversidad
+------------+---------------+
             |
             v
+----------------------------+
| 4. Planificación           |  Greedy NN + slots horarios + apertura
+------------+---------------+
             |
             v
+----------------------------+
| 5. Generación narrativa    |  Ollama (local) -> itinerario en español
+------------+---------------+
             |
             v
+----------------------------+
| 6. Evaluación automática   |  6 métricas objetivas con definición operativa
+----------------------------+
```

---

## 2. Arquitectura del sistema

### Módulo 1 — Intérprete de preferencias

Convierte texto libre a un objeto `UserPreferences` usando el LLM local (Ollama) con un prompt de sistema específico. Extrae: ámbito geográfico, duración, intereses, presupuesto, ritmo, movilidad, tipo de grupo y notas adicionales.

Cuando el usuario proporciona preferencias estructuradas directamente (formulario Streamlit), este módulo se omite y se usan los valores proporcionados directamente.

La extracción de JSON es robusta frente a texto extra que los modelos locales puedan añadir alrededor del objeto JSON (búsqueda de bloques markdown, búsqueda de llaves balanceadas).

### Módulo 2 — Recuperación semántica (RAG)

El `SemanticRetriever` construye una consulta enriquecida expandiendo cada interés a términos semánticos relacionados (ej. "museos" → "museo arte exposición colección cultural") y la embebe con **BAAI/bge-m3** (bi-encoder multilingüe).

La búsqueda vectorial en ChromaDB recupera los `k` POIs más similares semánticamente. Se aplican **filtros duros** post-recuperación: accesibilidad y ámbito geográfico.

Cada POI está indexado por su `enriched_text`, un texto denso en semántica que combina categoría, subcategoría, descripción, tags y contexto turístico adicional, diseñado explícitamente para maximizar la recuperación semántica.

### Módulo 3 — Reranking compuesto

El `POIRanker` combina tres señales de relevancia mediante pesos configurables en `config.json`:

| Señal | Peso | Descripción |
|-------|------|-------------|
| Semantic score | 0.30 | Similitud coseno del bi-encoder (ChromaDB, min-max normalizado) |
| Cross-encoder | 0.35 | Score de `ms-marco-multilingual-MiniLM-L12-v2` (sigmoide normalizado) |
| Preference match | 0.25 | Coincidencia entre categorías del POI e intereses del usuario |
| Diversity penalty | 0.10 | Penalización por repetición de categoría en el itinerario |

El cross-encoder analiza conjuntamente la consulta y el texto de cada POI (en lugar de por separado), ofreciendo mayor precisión aunque requiriendo más cómputo.

### Módulo 4 — Planificador de itinerarios

El `ItineraryPlanner` organiza los POIs en el itinerario mediante:

1. **Selección diaria**: elige N POIs/día según ritmo (tranquilo=3, moderado=4, intenso=6), validando apertura ese día de la semana y presupuesto acumulado.
2. **Clustering geográfico**: ordena los POIs de cada día por proximidad geográfica usando el algoritmo greedy del vecino más próximo con distancia Haversine.
3. **Asignación de franjas horarias**: mañana (09:30–14:00) y tarde (16:00–20:00), calculando tiempos de desplazamiento a pie entre POIs consecutivos.

### Módulo 5 — Generador narrativo

El LLM local (Ollama) recibe el itinerario estructurado (POIs, horarios, costes) junto con el perfil del viajero y genera un texto narrativo en español de estilo guía de viaje: amigable, informativo, con consejos prácticos y transiciones fluidas entre lugares.

Si Ollama no está disponible, el sistema genera automáticamente una narrativa básica de fallback con los datos estructurados, permitiendo que el resto del pipeline funcione igualmente.

### Módulo 6 — Evaluador automático

Calcula 6 métricas objetivas con definición operativa precisa (ver Sección 9). Las métricas se calculan sin intervención humana y se devuelven junto con la ruta en la respuesta de la API.

---

## 3. Tecnologías y modelos

| Componente | Tecnología | Justificación |
|-----------|-----------|---------------|
| API REST | FastAPI + Uvicorn | Alto rendimiento, validación automática con Pydantic v2 |
| Embeddings | `BAAI/bge-m3` | SOTA en recuperación multilingüe MTEB 2024, soporta español |
| Reranker | `ms-marco-multilingual-MiniLM-L12-v2` | Cross-encoder preciso para reranking en múltiples idiomas |
| Vector store | ChromaDB | Persistente, open-source, filtrado por metadatos |
| LLM | Ollama (local) | Inferencia 100% local, sin coste, sin dependencias externas |
| Frontend | Streamlit | Prototipado rápido con visualización interactiva |
| Mapa interactivo | Folium + streamlit-folium | Visualización de rutas con marcadores y polilíneas |

**Por qué BAAI/bge-m3:**
Modelo SOTA en benchmarks MTEB 2024 para recuperación semántica multilingüe (incluido español). Soporta hasta 8.192 tokens de contexto y embeddings de 1.024 dimensiones. Referencia: Chen et al. (2024).

**Por qué ms-marco-multilingual-MiniLM-L12-v2:**
Cross-encoder entrenado en datos MS MARCO multilingüe. Analiza conjuntamente la consulta y el documento, ofreciendo mayor precisión que el bi-encoder para el paso de reranking final al coste de mayor cómputo. Referencia: Nogueira & Cho (2019).

**Por qué Ollama:**
Permite correr modelos LLM de forma local sin necesidad de claves de API ni conexión a internet. Expone una API compatible con OpenAI, lo que facilita la integración. Modelos recomendados: `llama3.2` (por defecto), `mistral`, `qwen2.5`.

---

## 4. Estructura del proyecto

```
Proyecto/
|
+-- README.md
+-- config.json                    # Configuración global del sistema
+-- requirements.txt               # Dependencias Python
+-- Dockerfile
+-- docker-compose.yml             # Backend + frontend (Ollama corre en el host)
+-- entrypoint.sh                  # Script de arranque del contenedor
+-- prefetch_models.py             # Descarga modelos HuggingFace si no están en caché
|
+-- data/
|   +-- pois_bilbao_bizkaia.json  # 40 POIs de Bilbao y Bizkaia
|
+-- app/                           # Backend FastAPI
|   +-- main.py                    # Endpoints REST + ciclo de vida
|   +-- config.py                  # Cargador de configuración JSON
|   +-- models.py                  # Modelos Pydantic (POI, UserPreferences, TouristRoute...)
|   +-- interfaces.py              # Abstracciones (EmbeddingClient, VectorIndex)
|   +-- poi_manager.py             # Carga, indexación y consulta de POIs
|   +-- retriever.py               # Recuperación semántica RAG (SemanticRetriever)
|   +-- ranker.py                  # Reranking compuesto (POIRanker)
|   +-- planner.py                 # Planificación geográfica (ItineraryPlanner)
|   +-- generator.py               # Generación narrativa y ensamblado de ruta
|   +-- evaluator.py               # Métricas de evaluación automática
|   +-- infra/
|       +-- embeddings_local.py    # BAAI/bge-m3 via SentenceTransformers
|       +-- vector_chroma.py       # ChromaDB con normalización min-max de scores
|
+-- frontend/
|   +-- app.py                     # Interfaz Streamlit con 3 páginas
|
+-- db/                            # ChromaDB persistente (generado automáticamente)
+-- models_cache/                  # Cache de modelos HuggingFace (generado automáticamente)
```

---

## 5. Instalación

### Requisitos previos

- Python 3.10 o superior
- [Ollama](https://ollama.com) instalado y corriendo localmente
- Aprox. 4 GB de espacio en disco (modelos HuggingFace) + espacio para el modelo LLM (~2–4 GB según modelo)
- 8 GB de RAM recomendados (16 GB si se corre Ollama y los modelos HuggingFace simultáneamente)

### 1. Instalar y configurar Ollama

```bash
# Instalar desde https://ollama.com (Windows/Mac/Linux)

# Descargar el modelo LLM (una sola vez, ~2 GB)
ollama pull llama3.2

# Verificar que Ollama está corriendo
ollama list
```

Otros modelos compatibles (mejor calidad, más RAM):
```bash
ollama pull mistral          # ~4 GB, muy buena calidad en español
ollama pull qwen2.5          # ~4 GB, excelente multilingüe
ollama pull llama3.1:8b      # ~5 GB, más capaz
```

### 2. Instalación local del proyecto

```bash
# 1. Acceder al directorio del proyecto
cd "C:\Users\unaid\Documents\Master\NLP\Proyecto"

# 2. Crear entorno virtual
python -m venv .venv

# Activar en Windows
.venv\Scripts\activate

# Activar en Linux/Mac
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

#### Uso en local sin Docker

En local Ollama funciona directamente, no necesitas configuración adicional.

#### Uso con Docker — exponer Ollama al contenedor

Cuando se usa Docker, el backend corre dentro de un contenedor y necesita llegar a Ollama, que está en tu máquina. Por defecto Ollama escucha solo en `127.0.0.1` (no accesible desde Docker). Debes arrancarlo en todas las interfaces:

**Windows (PowerShell):**
```powershell
$env:OLLAMA_HOST='0.0.0.0:11434'; ollama serve
```

**Windows (cmd.exe):**
```batch
set OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

**Linux/Mac:**
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Sabrás que está bien cuando veas en la consola:
```
OLLAMA_HOST: http://0.0.0.0:11434
Listening on [::]:11434
```

> En Linux puedes hacer permanente la variable añadiendo `export OLLAMA_HOST=0.0.0.0:11434` a tu `~/.bashrc` o `~/.zshrc`.

El `docker-compose.yml` ya está configurado para usar `host.docker.internal:11434` (la dirección del host desde dentro de un contenedor), por lo que no necesitas tocar ningún otro fichero.

### Con Docker Compose

Ollama corre en tu máquina (ver paso anterior). Docker Compose solo levanta el backend y el frontend:

```bash
docker-compose up --build
```

- Backend API: http://localhost:8000
- Frontend Streamlit: http://localhost:8501

---

## 6. Configuración

Edita `config.json` antes de arrancar el sistema. Los campos más importantes:

```json
{
  "llm": {
    "ollama_base_url": "http://localhost:11434",
    "ollama_model_name": "llama3.2",
    "temperature_generation": 0.5,
    "temperature_interpretation": 0.1,
    "max_tokens_generation": 2000
  },
  "reranker": {
    "model_name": "cross-encoder/ms-marco-multilingual-MiniLM-L12-v2",
    "enabled": true
  },
  "rag": {
    "retrieval_k": 20,
    "rerank_top_n": 12
  },
  "planner": {
    "pois_per_day": {
      "tranquilo": 3,
      "moderado": 4,
      "intenso": 6
    }
  },
  "scoring_weights": {
    "semantic": 0.30,
    "rerank": 0.35,
    "preference_match": 0.25,
    "spatial_diversity": 0.10
  }
}
```

**Sin Ollama corriendo:** El sistema funciona parcialmente. El retriever, ranker y planificador funcionan sin LLM. La interpretación de texto libre y la narrativa usarán un fallback básico en texto plano con un aviso visible.

**Sin suficiente RAM:** Pon `"enabled": false` en la sección `reranker` para desactivar el cross-encoder. El sistema usará sólo los scores semánticos del bi-encoder.

**Cambiar modelo LLM:** Simplemente modifica `ollama_model_name` en `config.json` y asegúrate de haber hecho `ollama pull <nombre>` previamente.

---

## 7. Ejecución

Hay dos formas de ejecutar el sistema: **local** (desarrollo) o **Docker** (recomendado para despliegue).

---

### Opción A — Ejecución local (desarrollo)

#### Paso 1 — Ollama

En Windows, Ollama arranca automáticamente como servicio tras la instalación. Verifica que está corriendo:

```bash
ollama list
```

Si no está corriendo:
```bash
ollama serve
```

#### Paso 2 — Arrancar el backend

```bash
# Desde el directorio del proyecto, con el entorno virtual activado
uvicorn app.main:app --reload --port 8000
```

La primera vez descarga los modelos HuggingFace (~2 GB). Salida esperada:

```
[prefetch] Descargando modelos en models_cache/ ...
[prefetch]   → BAAI/bge-m3
[prefetch]   → cross-encoder/ms-marco-multilingual-MiniLM-L12-v2
[prefetch] ✅ Modelos listos.
INFO — 40 POIs cargados y listos.
INFO — Sistema listo en X.Xs.
```

Las siguientes ejecuciones detectan los modelos en caché y arrancan en segundos:
```
[prefetch] ✅ Modelos ya en caché. Saltando descarga.
INFO — Sistema listo en X.Xs.
```

#### Paso 3 — Arrancar el frontend (otra terminal)

```bash
streamlit run frontend/app.py
```

Abre http://localhost:8501 en tu navegador.

---

### Opción B — Docker Compose (recomendado)

#### Paso 1 — Ollama con acceso desde contenedores

Docker no puede acceder a `localhost` del host. Debes arrancar Ollama escuchando en todas las interfaces:

**Windows (PowerShell):**
```powershell
$env:OLLAMA_HOST='0.0.0.0:11434'; ollama serve
```

**Windows (cmd.exe):**
```batch
set OLLAMA_HOST=0.0.0.0:11434 && ollama serve
```

**Linux/Mac:**
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Confirma que está bien cuando veas: `Listening on [::]:11434`

#### Paso 2 — Primera ejecución (build + descarga de modelos)

```bash
docker-compose up --build
```

El contenedor ejecuta automáticamente `prefetch_models.py` antes de arrancar la API:

```
turismo_api  | [prefetch] Descargando modelos en /app/models_cache ...
turismo_api  |   → BAAI/bge-m3            (~1.5 GB, puede tardar varios minutos)
turismo_api  |   → cross-encoder/...      (~300 MB)
turismo_api  | [prefetch] ✅ Modelos listos. Caché guardada en volumen.
turismo_api  | INFO — Sistema listo en X.Xs.
```

Los modelos se guardan en el volumen `./models_cache/`. Si paras y reinicias, no se vuelven a descargar.

#### Paso 3 — Ejecuciones siguientes

```bash
docker-compose up
```

Sin `--build` si el código no ha cambiado. El arranque detecta la caché y salta la descarga:

```
turismo_api  | [prefetch] ✅ Modelos ya en caché. Saltando descarga.
turismo_api  | INFO — Sistema listo en X.Xs.
```

#### URLs

| Servicio | URL |
|----------|-----|
| Frontend Streamlit | http://localhost:9001 |
| API REST | http://localhost:9000 |
| Swagger / docs | http://localhost:9000/docs |

#### Forzar re-descarga de modelos

Si quieres que los modelos se vuelvan a descargar (versión nueva, caché corrupta):

```bash
# Borrar el marcador de caché
del models_cache\.models_ready        # Windows
rm models_cache/.models_ready         # Linux/Mac

# Reiniciar (descargará de nuevo automáticamente)
docker-compose up
```

O para borrar toda la caché:
```bash
rmdir /s /q models_cache   # Windows
rm -rf models_cache/       # Linux/Mac
docker-compose up
```

---

### Notas sobre puertos en Windows

Windows (Hyper-V/WSL2) reserva rangos de puertos que Docker no puede usar. Si ves el error `bind: Intento de acceso a un socket no permitido`, comprueba los puertos excluidos con:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

Los puertos configurados actualmente (9000 y 9001) están fuera de los rangos reservados habituales. Si necesitas cambiarlos, edita `docker-compose.yml`.

### Uso via API REST (curl)

```bash
# Ruta por texto libre
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{"query": "2 dias en Bilbao con mi pareja, museos y pintxos, presupuesto 60 euros/dia"}'

# Ruta con preferencias estructuradas
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "preferences": {
      "city_scope": "Bilbao",
      "duration_days": 2,
      "interests": ["museos", "gastronomia", "arquitectura"],
      "budget_per_day": 60.0,
      "pace": "moderado",
      "group_type": "pareja"
    }
  }'

# Busqueda semantica de POIs
curl -X POST http://localhost:8000/api/pois/search \
  -H "Content-Type: application/json" \
  -d '{"query": "playas con surf en la costa vasca", "k": 5}'

# Estado del sistema
curl http://localhost:8000/api/health
```

---

## 8. API REST

Documentación Swagger interactiva disponible en http://localhost:8000/docs

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/route` | Genera una ruta turística completa |
| `GET` | `/api/pois` | Lista todos los POIs (filtros: `category`, `municipality`) |
| `GET` | `/api/pois/{poi_id}` | Detalle completo de un POI |
| `POST` | `/api/pois/search` | Búsqueda semántica libre sobre la colección |
| `GET` | `/api/health` | Estado del sistema y modelos cargados |
| `GET` | `/api/stats` | Estadísticas de la colección de POIs |
| `POST` | `/api/admin/reindex` | Re-indexa todos los POIs en ChromaDB |

### Campos de respuesta de `/api/route`

```json
{
  "route": {
    "title": "Ruta de 2 dias por Bilbao: museos y gastronomia",
    "days": [
      {
        "day": 1,
        "pois": [
          {
            "poi": { "name": "Museo Guggenheim Bilbao", "..." : "..." },
            "slot": "manana",
            "start_time": "09:30",
            "end_time": "12:00",
            "semantic_score": 0.82,
            "rerank_score": 0.79,
            "final_score": 0.82
          }
        ],
        "total_cost_eur": 28.0
      }
    ],
    "narrative": "Bienvenidos a Bilbao...",
    "total_pois": 8,
    "total_cost_eur": 56.0
  },
  "evaluation": {
    "preference_coverage": 0.875,
    "temporal_coherence": 1.0,
    "geographic_consistency": 0.91,
    "budget_adherence": 1.0,
    "category_diversity": 0.75,
    "accessibility_compliance": 1.0,
    "overall_score": 0.91
  },
  "retrieval_info": {
    "candidates_retrieved": 20,
    "candidates_after_rerank": 12,
    "reranker_used": true,
    "embedding_model": "BAAI/bge-m3",
    "reranker_model": "cross-encoder/ms-marco-multilingual-MiniLM-L12-v2",
    "llm_model": "llama3.2"
  },
  "execution_time_seconds": 9.2
}
```

---

## 9. Métricas de evaluación

Las métricas se calculan automáticamente para cada ruta generada y se incluyen en la respuesta de la API. Todas devuelven valores en **[0, 1]** (1 = resultado óptimo).

| Métrica | Definición operativa | Peso |
|---------|---------------------|------|
| **Cobertura de preferencias** | `\|POIs con >=1 tag/categoría coincidente con interés del usuario\| / \|total POIs\|` | 0.25 |
| **Coherencia temporal** | `\|POIs cuyo horario de apertura cubre la franja horaria asignada\| / \|total POIs\|` | 0.25 |
| **Consistencia geográfica** | `mean_dias(max(0, 1 - dist_media_Haversine_km / 20))` — compacidad diaria | 0.20 |
| **Ajuste al presupuesto** | `mean_dias(max(0, 1 - max(0, coste_dia - presupuesto) / presupuesto))` | 0.15 |
| **Diversidad de categorías** | `\|categorías únicas en la ruta\| / \|total POIs\|` | 0.10 |
| **Accesibilidad** | Si `mobility=reducida`: `\|POIs accesibles\| / \|total POIs\|`; si `normal`: 1.0 | 0.05 |
| **Puntuación global** | Media ponderada de las seis métricas anteriores | — |

---

## 10. Corpus de datos

`data/pois_bilbao_bizkaia.json` contiene **40 Puntos de Interés** de Bilbao y Bizkaia recopilados de fuentes oficiales y abiertas.

### Fuentes de datos

| Fuente | Tipo de recurso | Uso previsto |
|--------|----------------|--------------|
| Open Data Euskadi | Dataset georreferenciado oficial | Base de POIs: nombre, tipo, localización |
| Bilbao Turismo | Web turística oficial | Horarios, precios, categorías, descripciones |
| Visit Biscay / Turismo Bizkaia | Web turística provincial | POIs fuera de Bilbao ciudad |
| OpenStreetMap | Base geoespacial abierta | Coordenadas y validación geográfica |
| Wikidata | Base de conocimiento abierta | Enlazado semántico |

### Campos de cada POI

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | string | Identificador único (`poi_001`...`poi_040`) |
| `name` | string | Nombre del punto de interés |
| `municipality` | string | Municipio (Bilbao, Getxo, Bermeo...) |
| `category` | string | Categoría principal |
| `subcategory` | string | Subcategoría específica |
| `description` | string | Descripción detallada |
| `coordinates` | {lat, lon} | Coordenadas WGS84 |
| `address` | string | Dirección postal |
| `price` | string | Rango de precio (gratis, EUR, EUR EUR, EUR EUR EUR) |
| `price_numeric` | float | Precio de entrada en euros |
| `schedule` | dict | Horario por día de la semana (null = cerrado) |
| `source` | string | Fuente de los datos |
| `url` | string | URL de referencia oficial |
| `tags` | list[str] | Etiquetas semánticas para la indexación |
| `enriched_text` | string | Texto enriquecido para indexación vectorial |
| `visit_duration_minutes` | int | Tiempo de visita estimado en minutos |
| `accessibility` | bool | Accesible para personas con movilidad reducida |

### Municipios cubiertos

Bilbao · Getxo · Bermeo · Sopelana · Mundaka · Gernika-Lumo · Kortezubi · Ibarrangelu · Lekeitio · Ondarroa · Bakio · Gatika · Balmaseda · Durango · Abadiano · Abanto-Zierbena

---

## 11. Mejoras respecto a la versión anterior

### Migración a Ollama (inferencia local)

- **Eliminada la dependencia de OpenAI**: ya no se necesita API key ni conexión a internet en tiempo de ejecución.
- **API compatible**: se mantiene la librería `openai` apuntando a `http://localhost:11434/v1`, lo que permite cambiar el modelo LLM simplemente editando `config.json` y haciendo `ollama pull <modelo>`.
- **Modelos recomendados**: `llama3.2` (por defecto, ~2 GB), `mistral` (~4 GB, mejor calidad en español), `qwen2.5` (~4 GB, excelente multilingüe).

### Extracción de JSON más robusta

Los modelos LLM locales pueden añadir texto introductorio o bloques markdown alrededor del JSON pedido. La función `_extract_json` en `generator.py` aplica tres estrategias en cascada:
1. Parseo directo del texto.
2. Extracción desde bloque ` ```json ... ``` `.
3. Búsqueda del primer objeto JSON con llaves balanceadas.

Esto elimina los errores de interpretación que antes fallaban silenciosamente con valores por defecto.

### Fallback con aviso explícito

Cuando Ollama no está disponible, el mensaje de fallback ahora incluye un aviso visible al usuario indicando que la narrativa es básica y cómo arrancar Ollama, en lugar de devolver texto plano sin contexto.

### Docker Compose con Ollama

El `docker-compose.yml` incluye ahora el servicio `ollama/ollama:latest` con volumen persistente para los modelos descargados. Soporte opcional para GPU NVIDIA documentado como comentario.

---

## 12. Referencias

- Lewis, P., Perez, E., Piktus, A., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *Advances in Neural Information Processing Systems (NeurIPS 2020)*. arXiv:2005.11401.

- Reimers, N. & Gurevych, I. (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**. *Proceedings of EMNLP 2019*. arXiv:1908.10084.

- Chen, J., Xiao, S., Zhang, P., et al. (2024). **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**. arXiv:2402.03216.

- Asai, A., Wu, Z., Wang, Y., et al. (2023). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**. arXiv:2310.11511.

- Guu, K., Lee, K., Tung, Z., et al. (2020). **REALM: Retrieval-Augmented Language Model Pre-Training**. *Proceedings of ICML 2020*. arXiv:2002.08909.

- Nogueira, R. & Cho, K. (2019). **Passage Re-ranking with BERT**. arXiv:1901.04085.

---

*Proyecto desarrollado para la asignatura de Procesamiento del Lenguaje Natural.*
*Máster Universitario en Inteligencia Artificial — 2025/2026.*
