"""
Métricas estándar de Information Retrieval implementadas desde cero.

Todas las funciones reciben listas de IDs ordenadas por score descendente.
No dependen de sklearn ni de ninguna librería de ranking externa.

Referencias:
  - Manning et al. (2008) "Introduction to Information Retrieval", cap. 8.
  - Järvelin & Kekäläinen (2002) "Cumulated gain-based evaluation of IR techniques", TOIS.
"""

import math
from typing import List, Set


def _relevant_set(relevant_ids: List[str]) -> Set[str]:
    return set(relevant_ids)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Proporción de relevantes recuperados entre los top-k resultados.

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|
    Devuelve 0.0 si relevant está vacío.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    rel = _relevant_set(relevant_ids)
    return len(top_k & rel) / len(rel)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Proporción de relevantes entre los top-k resultados.

    Precision@k = |retrieved[:k] ∩ relevant| / k
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    rel = _relevant_set(relevant_ids)
    hits = sum(1 for r in top_k if r in rel)
    return hits / k


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank para una sola query.

    MRR = 1 / rank_of_first_relevant  (0.0 si no hay ningún relevante en la lista)
    """
    rel = _relevant_set(relevant_ids)
    for rank, poi_id in enumerate(retrieved_ids, start=1):
        if poi_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    highly_relevant_ids: List[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain con relevancia graduada.

    Grados de relevancia:
      highly_relevant → 2
      relevant (pero no highly) → 1
      no relevante → 0

    Devuelve 0.0 si el IDCG es 0 (no hay relevantes para la query).
    """
    highly = set(highly_relevant_ids)
    rel = _relevant_set(relevant_ids) - highly  # solo los relevantes que no son highly

    def gain(poi_id: str) -> int:
        if poi_id in highly:
            return 2
        if poi_id in rel:
            return 1
        return 0

    # DCG del ranking recuperado
    dcg = 0.0
    for i, poi_id in enumerate(retrieved_ids[:k], start=1):
        dcg += gain(poi_id) / math.log2(i + 1)

    # IDCG: ranking ideal (todos los highly primero, luego los relevant)
    ideal_gains = sorted([2] * len(highly) + [1] * len(rel), reverse=True)
    idcg = 0.0
    for i, g in enumerate(ideal_gains[:k], start=1):
        idcg += g / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Average Precision para una sola query.

    AP = (1 / |relevant|) * sum_k [ Precision@k * rel(k) ]
    donde rel(k) = 1 si el k-ésimo resultado es relevante, 0 si no.
    """
    if not relevant_ids:
        return 0.0
    rel = _relevant_set(relevant_ids)
    hits = 0
    sum_prec = 0.0
    for rank, poi_id in enumerate(retrieved_ids, start=1):
        if poi_id in rel:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(rel)


def compute_all(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    highly_relevant_ids: List[str],
    ks: List[int] = None,
) -> dict:
    """Calcula todas las métricas para una query y devuelve un dict.

    Args:
        retrieved_ids:      IDs devueltos por el retriever, ordenados por score.
        relevant_ids:       IDs de la query en el gold set (relevant + highly_relevant).
        highly_relevant_ids: Subconjunto highly relevant.
        ks:                 Valores de k para Recall@k y Precision@k.

    Returns:
        Dict con claves: recall@k, precision@k (para cada k), mrr, ndcg@10, ap.
    """
    if ks is None:
        ks = [5, 10, 20]

    # highly_relevant también cuenta como relevant
    all_relevant = list(set(relevant_ids) | set(highly_relevant_ids))

    result = {}
    for k in ks:
        result[f"recall@{k}"]    = recall_at_k(retrieved_ids, all_relevant, k)
        result[f"precision@{k}"] = precision_at_k(retrieved_ids, all_relevant, k)

    result["mrr"]      = mrr(retrieved_ids, all_relevant)
    result["ndcg@10"]  = ndcg_at_k(retrieved_ids, all_relevant, highly_relevant_ids, 10)
    result["ap"]       = average_precision(retrieved_ids, all_relevant)
    return result
