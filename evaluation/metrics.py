import math
from typing import List, Set


def _relevant_set(relevant_ids: List[str]) -> Set[str]:
    return set(relevant_ids)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@k = |retrieved[:k] ∩ relevant| / |relevant|."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    rel = _relevant_set(relevant_ids)
    return len(top_k & rel) / len(rel)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Precision@k = |retrieved[:k] ∩ relevant| / k."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    rel = _relevant_set(relevant_ids)
    hits = sum(1 for r in top_k if r in rel)
    return hits / k


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR = 1 / rank del primer relevante. 0.0 si no hay ninguno."""
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
    """NDCG@k con relevancia graduada: highly=2, relevant=1, no=0."""
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
    """AP = (1/|relevant|) * sum_k Precision@k * rel(k)."""
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
    """Calcula Recall@k, Precision@k, MRR, NDCG@10 y AP en un solo dict."""
    if ks is None:
        ks = [5, 10, 20]

    all_relevant = list(set(relevant_ids) | set(highly_relevant_ids))

    result = {}
    for k in ks:
        result[f"recall@{k}"]    = recall_at_k(retrieved_ids, all_relevant, k)
        result[f"precision@{k}"] = precision_at_k(retrieved_ids, all_relevant, k)

    result["mrr"]      = mrr(retrieved_ids, all_relevant)
    result["ndcg@10"]  = ndcg_at_k(retrieved_ids, all_relevant, highly_relevant_ids, 10)
    result["ap"]       = average_precision(retrieved_ids, all_relevant)
    return result
