# Roadmap pendiente — Hito 2

Listado de tareas pendientes para cerrar el Hito 2 con rigor académico suficiente para la
defensa de la memoria. Orden por prioridad descendente.

---

## 🔴 Alta prioridad — Bloquean la memoria académica

### P1 — Ejecutar el baseline del retriever denso en el entorno Docker

**Estado:** El script `evaluation/eval_retrieval.py` está implementado y testeado, pero
el entorno local de Windows tiene conflictos de dependencias (`chromadb 0.6.x` vs.
`huggingface_hub` vs. `transformers`). En Docker no hay este problema.

**Acción:**
```bash
docker-compose run --rm api python -m evaluation.eval_retrieval \
    --gold evaluation/gold_set.json \
    --mode dense --k 20 \
    --output results/baseline_dense.json

docker-compose run --rm api python -m evaluation.eval_retrieval \
    --gold evaluation/gold_set.json \
    --mode dense --with-reranker --k 20 \
    --output results/baseline_dense_reranker.json
```

Rellenar `results/baseline_dense.md` con los números reales. Sin estos números no hay
sección "Resultados" en la memoria.

---

### P2 — TAREA 2 completa: BM25 + Reciprocal Rank Fusion

**Justificación:** El README menciona "RAG híbrido" pero el sistema actual es solo
denso. Sin BM25 no podemos defender el adjetivo "híbrido" en el tribunal.

**Subtareas:**

#### P2.1 — `app/infra/bm25_index.py`
- Dependencia nueva: `rank_bm25>=0.2.2` en `requirements.txt`.
- Tokenizer simple español (lowercase + split + ~150 stopwords castellanas inline).
- Parámetros BM25: `k1=1.5, b=0.75` (estándar TREC).
- Persistencia en `db/bm25.pkl` para no reconstruir en cada arranque.
- API: `build()`, `search(query, k)`, `persist(path)`, `load(path)`.

#### P2.2 — `app/hybrid_retriever.py`
- Fusión por **Reciprocal Rank Fusion** (Cormack et al. 2009, TREC):
  `score(poi) = sum_over_retrievers(1 / (rrf_k + rank_in_retriever))` con `rrf_k=60`.
- Alternativa configurable: combinación lineal `α·dense + (1-α)·bm25_normalized`.
- Aplica los mismos filtros duros que `SemanticRetriever` (accesibilidad, scope).

#### P2.3 — Integración
- Añadir bloque `retrieval` a `config.json`:
  ```json
  "retrieval": {
    "mode": "hybrid",
    "fusion": "rrf",
    "rrf_k": 60,
    "linear_alpha": 0.5
  }
  ```
- Modificar `app/main.py` lifespan para construir BM25Index y HybridRetriever.

#### P2.4 — Completar evaluador y generar ablation
```bash
python -m evaluation.eval_retrieval --mode bm25 --output results/bm25.json
python -m evaluation.eval_retrieval --mode hybrid --output results/hybrid.json
python -m evaluation.eval_retrieval --mode hybrid --with-reranker --output results/hybrid_reranker.json
```
Rellenar `results/ablation_retrieval.md` con tabla final:

| Configuración          | Recall@5 | Recall@10 | MRR | NDCG@10 | MAP |
|------------------------|----------|-----------|-----|---------|-----|
| Dense (bge-m3)         | …        | …         | …   | …       | …   |
| BM25                   | …        | …         | …   | …       | …   |
| Hybrid (RRF)           | …        | …         | …   | …       | …   |
| Dense + Reranker       | …        | …         | …   | …       | …   |
| Hybrid + Reranker      | …        | …         | …   | …       | …   |

Discusión cualitativa: ¿qué tipo de query gana cada método?
- Paráfrasis (q021-q030) → previsiblemente dense
- Topónimos (q032 Getxo, q037 Gernika) → previsiblemente BM25
- Adversariales con negación (q031, q035) → ambos sufren

#### P2.5 — Tests
- `tests/test_bm25_index.py`: build, search por nombre exacto, persistencia.
- `tests/test_hybrid_retriever.py`: RRF con rankings sintéticos conocidos, filtros duros.

#### P2.6 — Actualizar README
Quitar las menciones falsas a "híbrido" del estado actual y describir la arquitectura real
una vez implementada la Tarea 2.

---

### P3 — Revisión humana del gold set

**Problema:** Las 40 queries fueron anotadas semi-automáticamente. Para defensa académica
hace falta revisión cruzada de **3 anotadores** (yo + Xabier + Oscar).

**Acción:**
- ~1h por persona revisando las 40 queries.
- Calcular **Cohen's κ** entre pares de anotadores (objetivo κ > 0.7).
- Discutir y consensuar las discrepancias.
- Versionar el gold set: `gold_set_v1.0.json` (auto), `gold_set_v1.1.json` (revisado).

---

## 🟡 Media prioridad — Mejoran rigor, no son bloqueantes

### P4 — FASE 3.1: Expansión semántica de queries con embeddings

**Problema:** `_build_query()` en `app/retriever.py` usa un mapeo fijo
`INTEREST_QUERY_MAP` (diccionario hardcoded). Es frágil: cualquier interés no listado
cae al literal. Una expansión semántica con embeddings sería más robusta.

**Propuesta:** Antes de construir la query, expandir cada interés con sus k vecinos más
cercanos en el espacio de embeddings de las categorías del corpus.

**Trade-off:** Aumenta latencia (k×embeddings extra) — quizás solo merece la pena para
intereses raros (`surf`, `senderismo`).

---

### P5 — FASE 4.1: TSP 2-opt en el planner

**Problema:** El planner usa Nearest-Neighbor greedy para ordenar POIs de un día.
NN es subóptimo en el peor caso ~25% peor que TSP óptimo.

**Propuesta:** Aplicar 2-opt sobre la solución NN como post-procesamiento.
Para N≤7 POIs/día el coste es despreciable (<10ms).

**Métrica de impacto:** Reducción esperada en `total_travel_minutes` por día.
Comparar antes/después en evaluación cuantitativa de itinerarios.

---

### P6 — FASE 4.2: Validación de `config.json` con Pydantic BaseSettings

**Problema:** `app/config.py` carga JSON crudo sin validación. Un typo en una clave
(`scoring_weights → "semantik"`) pasa silenciosamente y degrada el sistema.

**Propuesta:** Reemplazar `Settings` por una `BaseSettings` de Pydantic v2 con tipos
estrictos. Errores de configuración fallan al arrancar, no en runtime.

---

### P7 — FASE 4.3: Bug en filtro de gratuidad del retriever

**Problema:** En [app/retriever.py:168](app/retriever.py#L168), el descuento por precio
penaliza POIs caros, pero no hay bonificación inversa para POIs gratuitos cuando el
presupuesto es muy bajo. La asimetría está en el ranker pero no en el retriever, lo que
puede llevar a que POIs gratis no lleguen al ranker si el score semántico es bajo.

**Propuesta:** Bonificar el score (×1.1) para POIs gratuitos cuando `budget_per_day < 30`.

---

## 🟢 Baja prioridad — Pulido y memoria

### P8 — FASE 5: MEMORIA.md académica (6-8 páginas)

**Estructura propuesta:**
1. **Introducción** (0.5p): Problema, motivación, contribuciones.
2. **Estado del arte** (1p): RAG (Lewis et al. 2020), bge-m3, BM25, RRF.
3. **Arquitectura** (1.5p): Pipeline R→R→P→G, decisiones de diseño.
4. **Evaluación** (2p): Gold set, métricas, tabla ablation, análisis cualitativo.
5. **Discusión** (1p): Limitaciones, fortalezas, trade-offs.
6. **Conclusiones y trabajo futuro** (0.5p).
7. **Referencias** (0.5p).

**Bloqueado por:** P1, P2 (necesita las tablas reales).

---

### P9 — Ampliar el corpus gastronómico

**Problema:** El corpus solo tiene 5 POIs de `category=gastronomía` (todos mercados).
No hay restaurantes individuales ni bares de pintxos. Esto sesga las queries q003, q022,
q035 del gold set: no hay mucha variedad donde elegir.

**Propuesta:** Ejecutar `scripts/expand_bilbao_corpus.py` con un foco gastronómico o
añadir manualmente ~50 restaurantes/bares emblemáticos de Bilbao.

---

### P10 — Latencia y métricas de eficiencia

**Idea:** Reportar también latencia mediana de retrieval por modo en la tabla ablation.
El script ya guarda `latency_ms` por query — solo falta agregarlo.

Argumento de defensa: "Hybrid+Reranker da +5% NDCG pero a costa de 3× latencia". Esa es
una decisión arquitectural digna de discusión en la memoria.

---

### P11 — Pruebas E2E con FastAPI TestClient

**Estado:** Los tests cubren unidades (planner, evaluator, eval_retrieval, extract_json).
No hay tests de integración del endpoint `POST /api/route`.

**Propuesta:** Un `tests/test_api.py` con TestClient que dispare 3-5 requests con
preferencias diferentes y compruebe estructura de respuesta + métricas mínimas.

---

## Notas de proceso

- **Convención de commits:** Conventional Commits (`feat:`, `fix:`, `refactor:`,
  `test:`, `docs:`).
- **Antes de subir resultados a `results/`:** Verificar reproducibilidad ejecutando dos
  veces y comparando que las métricas son iguales (no debería haber estocasticidad).
- **Si añades dependencia nueva:** Actualizar `requirements.txt` + justificar en
  `MEJORAS.md`.
