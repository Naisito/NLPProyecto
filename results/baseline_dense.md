# Baseline — Dense Retrieval (BAAI/bge-m3)

## Cómo reproducir

El script requiere el entorno Docker del proyecto (ChromaDB indexado + modelos descargados).

```bash
# Desde dentro del contenedor o con el entorno de pip del proyecto:
python -m evaluation.eval_retrieval \
    --gold evaluation/gold_set.json \
    --mode dense \
    --k 20 \
    --output results/baseline_dense.json
```

## Tabla de resultados

> **Nota:** Pendiente de ejecutar en el entorno Docker.
> Reemplazar los valores de esta tabla con la salida real del script.

| Configuración       | Recall@5 | Recall@10 | MRR    | NDCG@10 | MAP    |
|---------------------|----------|-----------|--------|---------|--------|
| Dense (bge-m3)      | —        | —         | —      | —       | —      |
| Dense + Reranker    | —        | —         | —      | —       | —      |

## Análisis por grupo de queries

| Grupo             | Recall@10 esperado | Observación |
|-------------------|--------------------|-------------|
| interes_principal | Alto (~0.6–0.8)    | Queries directas; bge-m3 debería recuperar bien por semántica |
| multi_interes     | Medio-alto         | La consulta enriquecida une dos campos semánticos |
| restricciones     | Medio              | Depende de si el retriever aplica filtros duros antes de la métrica |
| paráfrasis        | Medio (~0.4–0.6)   | Test clave del RAG: sinónimos sin términos exactos |
| adversarial       | Bajo-Medio         | Negaciones y topónimos son el punto débil del retriever denso |

## Observaciones esperadas

- **Paráfrasis vs. adversarial**: El retriever denso (bge-m3 multilingüe) debería manejar bien las paráfrasis (q021–q030) por su capacidad de capturar similitud semántica. Las queries adversariales con negación (q031, q035) son el escenario más difícil.
- **Topónimos (q037 Gernika)**: BM25 superará al denso aquí — motivación principal para la Tarea 2.
- **Intereses sin categoría (q033 surf, q040 senderismo)**: El retriever denso debería recuperar playas/naturaleza por semántica aunque "surf" no exista como categoría explícita.

## Formato del JSON de salida (`results/baseline_dense.json`)

```json
{
  "mode": "dense",
  "with_reranker": false,
  "k": 20,
  "gold_set": "evaluation/gold_set.json",
  "n_queries": 40,
  "aggregate": {
    "recall@5":    0.XXXX,
    "recall@10":   0.XXXX,
    "recall@20":   0.XXXX,
    "precision@5": 0.XXXX,
    "precision@10":0.XXXX,
    "precision@20":0.XXXX,
    "mrr":         0.XXXX,
    "ndcg@10":     0.XXXX,
    "ap":          0.XXXX
  },
  "per_query": [...]
}
```
