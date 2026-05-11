# Evaluación cuantitativa del retriever

Scripts para medir el rendimiento de la fase de recuperación del sistema RAG.

## Estructura

```
evaluation/
├── gold_set.json       # 40 queries anotadas manualmente
├── metrics.py          # Funciones IR: Recall@k, Precision@k, MRR, NDCG@10, MAP
├── eval_retrieval.py   # CLI de evaluación
└── README.md           # Este archivo
results/
├── baseline_dense.md   # Tabla de resultados del retriever denso (bge-m3)
└── ablation_retrieval.md  # Tabla ablation completa (Tarea 2)
```

## Gold set

### Distribución de las 40 queries

| Grupo             | Queries | Descripción |
|-------------------|---------|-------------|
| interes_principal | 10      | Una categoría de interés simple (museos, playa, naturaleza…) |
| multi_interes     | 5       | Combinación de dos intereses |
| restricciones     | 5       | Accesibilidad, presupuesto bajo, familia, horario |
| paráfrasis        | 10      | Sinónimos y paráfrasis sin usar los términos exactos del corpus |
| adversarial       | 10      | Negaciones, topónimos, intereses sin categoría propia (surf) |

### Criterios de anotación

- **highly_relevant**: El POI es referencia canónica para esa query (ej. Guggenheim para arte contemporáneo).
- **relevant**: El POI pertenece a la categoría/municipio que la query implica.

La asignación se basa en **category, subcategory, municipality y tags** del JSON — NO en el `INTEREST_TO_CATEGORIES` del ranker para evitar circularidad.

### Revisar/ampliar el gold set

Para añadir una query nueva, añade un objeto al array `queries` en `gold_set.json` con estos campos:

```json
{
  "id": "q041",
  "group": "paráfrasis",
  "query": "Texto de la query en español",
  "preferences": {"city_scope": "Bilbao", "interests": ["museos"]},
  "relevant_poi_ids": ["poi_xxx", "poi_yyy"],
  "highly_relevant_poi_ids": ["poi_xxx"],
  "notes": "Explicación de por qué estos POIs son relevantes"
}
```

Los IDs de los POIs se pueden encontrar buscando en `data/pois_bilbao_bizkaia.json` por nombre, categoría o municipio.

## Cómo ejecutar la evaluación

### Requisitos previos

El servidor no necesita estar activo; el script carga los modelos directamente. Asegúrate de que el índice ChromaDB esté construido (`db/chroma_db/`).

### Modo dense (retriever semántico puro)

```bash
python -m evaluation.eval_retrieval \
    --gold evaluation/gold_set.json \
    --mode dense \
    --output results/dense.json
```

### Modo dense + reranker

```bash
python -m evaluation.eval_retrieval \
    --gold evaluation/gold_set.json \
    --mode dense \
    --with-reranker \
    --output results/dense_reranker.json
```

### Modos BM25 e Híbrido (Tarea 2)

```bash
# BM25 solo
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode bm25

# Híbrido (RRF por defecto)
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode hybrid
```

Estos modos requieren que `app/infra/bm25_index.py` y `app/hybrid_retriever.py` estén implementados.

### Número de resultados a recuperar

```bash
# Recuperar top-30 en lugar del default (20)
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode dense --k 30
```

## Métricas implementadas

| Métrica | Descripción |
|---------|-------------|
| Recall@k | \|retrieved[:k] ∩ relevant\| / \|relevant\| |
| Precision@k | \|retrieved[:k] ∩ relevant\| / k |
| MRR | 1 / posición del primer relevante |
| NDCG@10 | DCG normalizado con relevancia graduada (highly=2, relevant=1) |
| MAP | Media de Average Precision sobre todas las queries |

## Interpretar los resultados

- **Recall@10 > 0.5**: El sistema recupera más de la mitad de los relevantes en top-10.
- **MRR > 0.5**: El primer resultado relevante aparece, en promedio, en las 2 primeras posiciones.
- **NDCG@10 > MRR**: Indica que los highly_relevant aparecen bien posicionados.

Valores bajos en queries del grupo **adversarial** son esperables; son queries diseñadas para detectar limitaciones.

## Tests

```bash
pytest tests/test_eval_retrieval.py -v
```
