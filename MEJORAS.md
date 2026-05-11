# Registro de mejoras — Hito 2

Log de cambios aplicados para subir el rigor científico y corregir errores del Hito 1.
Cada entrada incluye el archivo afectado, el problema resuelto y la decisión tomada.

---

## FASE 0 — Corrección de bugs

### F0.1 — Bug en `_is_open_at` (`app/planner.py`)

**Problema:** La condición anterior permitía que un POI con horario 13:00-14:00 fuera
considerado "abierto" para un slot 09:30-14:00 con visita de 60 min, porque solo
comprobaba `open_min ≤ end_min AND close_min ≥ start_min + visit_duration`. Esto es
un error lógico: no verifica que la visita quepa dentro de la intersección real entre
el horario del POI y la franja horaria.

**Solución:**
```python
effective_start = max(open_min, start_min)
effective_end   = min(close_min, end_min)
return (effective_end - effective_start) >= poi.visit_duration_minutes
```

**Impacto:** Corrige la métrica `temporal_coherence` (antes inflada artificialmente).
Los 6 tests de `TestIsOpenAt` validan la regresión.

---

### F0.2 — Inconsistencia reranker (`app/ranker.py`)

**Problema:** `config.json` declara `BAAI/bge-reranker-v2-m3` pero el fallback en código
era `cross-encoder/ms-marco-multilingual-MiniLM-L12-v2`. Además, la normalización
sigmoide es correcta para MSMARCO (logits sin restricción de escala) pero no garantiza
discriminación adecuada para BGE (que puede producir rangos distintos).

**Decisión:** Mantener `BAAI/bge-reranker-v2-m3` como modelo oficial (más SOTA,
alineado con `bge-m3` del embedder). Sustituir sigmoid por **normalización min-max**
sobre el batch, que es agnóstica respecto a la escala de salida del modelo:

```python
min_s, max_s = min(raw_scores), max(raw_scores)
normalized = [(s - min_s) / (max_s - min_s) for s in raw_scores]
```

Si todos los scores son iguales (max-min < 1e-6) se devuelve 0.5 uniforme.

**También:** Se corrige el truncado del texto del POI: antes se truncaba a 512
*caracteres* (`enriched_text[:512]`), lo que desperdiciaba capacidad del cross-encoder
(512 *tokens* >> 512 chars). Ahora se pasa el texto completo y el `CrossEncoder` aplica
el truncado por tokens internamente vía `max_length=512`.

---

### F0.3 — Planner ignoraba `start_hour`/`end_hour` del usuario (`app/planner.py`)

**Problema:** El método `plan()` usaba siempre las franjas fijas del `config.json`
(09:30-14:00 / 16:00-20:00), ignorando `preferences.start_hour` y `preferences.end_hour`.
La métrica `time_window_match` del evaluador penalizaba las propias salidas del sistema.

**Solución:** Al inicio de cada día, se ajustan los límites de las franjas:
- Si la ventana del usuario es < 4h: un único slot continuo entre `user_start` y `user_end`.
- Si no: `manana_start = max(cfg_start, user_start)`, `tarde_end = min(cfg_end, user_end)`.

Criterio verificado: con `start_hour=11:00, end_hour=18:00` ningún POI empieza antes
de las 11:00 ni termina después de las 18:00. Tests en `TestPlannerTimeWindow`.

---

### F0.4 — Código muerto en `app/retriever.py`

**Problema:** El bloque `if poi.price_numeric > daily_budget * 0.8: pass` no tenía
efecto, contradiciendo el comentario que indicaba que "el ranker penalizará los caros".

**Solución:** Convertido en descuento real de score para que la información llegue al
ranker ya pre-filtrada:
```python
if poi.price_numeric > daily_budget * 0.8 and poi.price_numeric > 0:
    score *= 0.7
```

---

## FASE 1 — RAG real en la generación narrativa (`app/generator.py`)

**Problema:** El generador narrativo recibía el itinerario estructurado pero **no el
contexto recuperado** por ChromaDB. La "G" del RAG no consumía la "R": el LLM debía
confiar en su pretraining para describir cada POI, con riesgo de alucinaciones.

**Solución en dos partes:**

1. **`_format_day_for_prompt`**: Para cada POI se incluye ahora el bloque
   `CONTEXTO_RECUPERADO: <enriched_text[:400]>` en el prompt del usuario.

2. **`_GEN_SYSTEM`**: Se añade una sección de control de alucinaciones obligatoria:
   el LLM debe basar sus afirmaciones en el bloque `CONTEXTO_RECUPERADO` y
   **no inventar** fechas, arquitectos, precios o direcciones.

**Relevancia académica:** Este cambio convierte el sistema en un RAG real (retrieval
*augmented* generation, no solo retrieval *informed* generation). Es el argumento más
sólido para la sección "Arquitectura" de la memoria.

---

## FASE 3.2 — JSON mode + few-shot en `interpret_preferences` (`app/generator.py`)

**Problema:** La interpretación de preferencias dependía de la fiabilidad del LLM para
producir JSON bien formado. Cualquier texto introductorio o error de formato causaba
que `_extract_json` tuviera que aplicar heurísticas costosas.

**Solución:**
- Se añade `response_format={"type": "json_object"}` a la llamada Ollama (JSON mode
  garantizado, elimina texto introductorio y bloques markdown).
- Se añaden 3 *few-shot examples* al system prompt cubriendo: query corta ambigua,
  query larga con presupuesto explícito, y query con condicionantes (movilidad reducida
  + familia con end_hour restrictivo).

**Nota:** Ollama soporta JSON mode desde la versión 0.1.24. Si se usa una versión
anterior, el modo degradará silenciosamente a texto libre y `_extract_json` seguirá
funcionando como fallback.

---

## FASE 4.3 — Tests unitarios (`tests/`)

Creados con `pytest`:

| Archivo | Tests | Cubre |
|---|---|---|
| `test_planner.py` | 9 | `_is_open_at` (6), ventana horaria (2), NN (1) |
| `test_evaluator.py` | 16 | Una clase por métrica, fixtures sin LLM |
| `test_extract_json.py` | 5 | JSON limpio, `<think>`, markdown, texto extra, malformado |

Resultado: **30/30 en verde** (`pytest tests/ -v`).

---

## FASE 2 — Evaluación cuantitativa del retriever (Tarea 1)

### F2.1 — Gold set anotado (`evaluation/gold_set.json`)

**Problema:** No existía ninguna forma cuantitativa de medir la calidad de la fase de
recuperación. Las afirmaciones sobre el rendimiento del retriever eran solo cualitativas,
lo que es inadmisible para una memoria académica.

**Solución:** Construcción de un **gold set de 40 queries anotadas manualmente** sobre
el corpus de 1109 POIs, con distribución equilibrada por tipo de dificultad:

| Grupo               | Nº | Propósito |
|---------------------|----|-----------|
| `interes_principal` | 10 | Una categoría de interés simple (museos, playa, naturaleza, gastronomía) |
| `multi_interes`     | 5  | Combinación de dos intereses |
| `restricciones`     | 5  | Accesibilidad, presupuesto bajo, familia, horario tarde |
| `paráfrasis`        | 10 | Sinónimos sin términos exactos del corpus (test de robustez semántica) |
| `adversarial`       | 10 | Negaciones, topónimos (Gernika, Getxo), intereses sin categoría propia (surf) |

**Decisión metodológica clave:** La anotación se basa en `category`, `subcategory`,
`municipality` y `tags` del JSON original — **NO** en el `INTEREST_TO_CATEGORIES` del
ranker. Esto evita la circularidad metodológica que invalidaría las métricas.

Cada query incluye:
- `relevant_poi_ids`: POIs que pertenecen al tema (gain=1 en NDCG)
- `highly_relevant_poi_ids`: POIs de referencia canónica (gain=2 en NDCG)
- `notes`: Justificación de la anotación

### F2.2 — Métricas IR desde cero (`evaluation/metrics.py`)

**Problema:** Las librerías como `sklearn` no incluyen MRR/NDCG con relevancia graduada,
y otras dependencias (`pytrec_eval`) son pesadas y mal documentadas.

**Solución:** Implementación desde cero de todas las métricas estándar de IR:
- `recall_at_k(retrieved, relevant, k)`
- `precision_at_k(retrieved, relevant, k)`
- `mrr(retrieved, relevant)` — Mean Reciprocal Rank
- `ndcg_at_k(retrieved, relevant, highly_relevant, k)` — con relevancia graduada (2/1/0)
- `average_precision(retrieved, relevant)` — base de MAP

**Referencias** documentadas en docstrings:
- Manning et al. (2008) "Introduction to Information Retrieval", cap. 8.
- Järvelin & Kekäläinen (2002) "Cumulated gain-based evaluation of IR techniques", TOIS.

### F2.3 — CLI de evaluación (`evaluation/eval_retrieval.py`)

**API:**
```bash
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode dense
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode dense --with-reranker
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode bm25
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode hybrid
```

**Salida:** Tabla markdown imprimible + JSON con métricas agregadas + métricas por query
+ top-10 IDs recuperados + latencia por consulta.

**Diseño:** Reutiliza `SemanticRetriever.search_by_text` y `POIRanker.rank` sin tocar
ningún módulo de `app/`. Los modos `bm25` e `hybrid` lanzan `NotImplementedError`
limpio como stubs preparados para la Tarea 2.

### F2.4 — Tests del evaluador (`tests/test_eval_retrieval.py`)

29 tests nuevos cubriendo:
- `recall_at_k`: 6 casos (perfecto, vacío, parcial, k>retrieved, etc.)
- `precision_at_k`: 4 casos
- `mrr`: 4 casos (primer relevante en pos 1, 3, no presente, múltiples)
- `ndcg_at_k`: 5 casos (relevancia graduada, orden subóptimo, sin highly)
- `average_precision`: 4 casos
- `compute_all`: 2 tests de integración
- `load_gold_set`: 4 tests de schema (válido, falta `queries`, falta campo, real)

Resultado total combinado: **43/43 tests en verde** (`pytest tests/ -v`).

### F2.5 — Documentación (`evaluation/README.md` + `results/baseline_dense.md`)

- `evaluation/README.md`: Estructura, cómo ejecutar, cómo añadir queries, métricas e interpretación.
- `results/baseline_dense.md`: Template con instrucciones, análisis por grupo de queries esperado,
  y formato del JSON de salida. Pendiente de rellenar al ejecutar en Docker.

**Relevancia académica:** Este bloque es el argumento más sólido para la sección
"Evaluación" de la memoria. Convierte el sistema de "creemos que funciona" a
"medimos con métricas estándar sobre un gold set propio".

---

## Pendiente

Ver [POR_MEJORAR.md](POR_MEJORAR.md) para el listado completo de tareas pendientes
con prioridad y justificación académica.
