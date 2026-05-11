import json
import math
import os
import tempfile

import pytest

from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    average_precision,
    compute_all,
)
from evaluation.eval_retrieval import load_gold_set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_ranking():
    """Todos los relevantes están en las primeras posiciones."""
    relevant = ["a", "b", "c"]
    retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    return retrieved, relevant


@pytest.fixture
def no_relevant_ranking():
    """Ningún relevante en los resultados."""
    relevant = ["x", "y", "z"]
    retrieved = ["a", "b", "c", "d", "e"]
    return retrieved, relevant


@pytest.fixture
def partial_ranking():
    """Solo la mitad de relevantes recuperados."""
    relevant = ["a", "b", "c", "d"]  # 4 relevantes
    retrieved = ["a", "x", "b", "y", "z"]  # 2 de 4 en top-5
    return retrieved, relevant


# ---------------------------------------------------------------------------
# Tests recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:

    def test_perfect_recall_at_3(self, perfect_ranking):
        retrieved, relevant = perfect_ranking
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_recall_zero_when_no_relevant(self, no_relevant_ranking):
        retrieved, relevant = no_relevant_ranking
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.0)

    def test_partial_recall_at_5(self, partial_ranking):
        retrieved, relevant = partial_ranking
        # a y b están en top-5 de 4 relevantes → 2/4 = 0.5
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)

    def test_recall_at_1_with_first_relevant(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=1) == pytest.approx(0.5)

    def test_recall_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=5) == pytest.approx(0.0)

    def test_recall_k_larger_than_retrieved(self):
        # k=20 pero solo 3 recuperados; deben contar todos los recuperados
        retrieved = ["a", "b", "x"]
        relevant = ["a", "b", "c", "d"]
        assert recall_at_k(retrieved, relevant, k=20) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:

    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_zero_precision(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b", "c"], k=3) == pytest.approx(0.0)

    def test_half_precision(self):
        # 1 de 2 en top-2
        assert precision_at_k(["a", "x"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_precision_k_zero(self):
        assert precision_at_k(["a", "b"], ["a"], k=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests mrr
# ---------------------------------------------------------------------------

class TestMRR:

    def test_first_relevant_at_position_1(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        assert mrr(retrieved, relevant) == pytest.approx(1.0)

    def test_first_relevant_at_position_3(self):
        retrieved = ["x", "y", "a", "b"]
        relevant = ["a"]
        assert mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_returns_zero(self, no_relevant_ranking):
        retrieved, relevant = no_relevant_ranking
        assert mrr(retrieved, relevant) == pytest.approx(0.0)

    def test_multiple_relevant_uses_first(self):
        retrieved = ["x", "a", "b", "c"]
        relevant = ["a", "b", "c"]
        # Primer relevante en pos 2 → MRR = 1/2
        assert mrr(retrieved, relevant) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests ndcg_at_k
# ---------------------------------------------------------------------------

class TestNDCGAtK:

    def test_perfect_ndcg(self):
        # highly=[a], relevant=[b], ideal=[2,1], retrieved=[a,b]
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b"]
        highly = ["a"]
        # IDCG = 2/log2(2) + 1/log2(3) = 2 + 0.6309 = 2.6309
        # DCG  = 2/log2(2) + 1/log2(3) = same → NDCG = 1.0
        assert ndcg_at_k(retrieved, relevant, highly, k=5) == pytest.approx(1.0)

    def test_zero_ndcg_no_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = []
        highly = []
        assert ndcg_at_k(retrieved, relevant, highly, k=5) == pytest.approx(0.0)

    def test_highly_relevant_before_relevant(self):
        # Si primero viene highly y luego relevant, NDCG debe ser máximo (1.0 para este caso)
        retrieved = ["h1", "r1", "x"]
        highly = ["h1"]
        relevant = ["r1"]
        assert ndcg_at_k(retrieved, relevant, highly, k=3) == pytest.approx(1.0)

    def test_relevant_before_highly_reduces_ndcg(self):
        # Orden subóptimo: relevant primero, highly después → NDCG < 1
        retrieved = ["r1", "h1", "x"]
        highly = ["h1"]
        relevant = ["r1"]
        score = ndcg_at_k(retrieved, relevant, highly, k=3)
        assert 0.0 < score < 1.0

    def test_no_highly_relevant_uses_only_relevant(self):
        # Si no hay highly, todos los relevantes tienen gain=1
        retrieved = ["a", "b", "x", "y"]
        relevant = ["a", "b"]
        highly = []
        assert ndcg_at_k(retrieved, relevant, highly, k=4) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests average_precision
# ---------------------------------------------------------------------------

class TestAP:

    def test_perfect_ap(self):
        # Todos relevantes al principio
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert average_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_ap_single_relevant_at_pos_1(self):
        retrieved = ["a", "x", "y"]
        relevant = ["a"]
        assert average_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_ap_single_relevant_at_pos_2(self):
        retrieved = ["x", "a", "y"]
        relevant = ["a"]
        assert average_precision(retrieved, relevant) == pytest.approx(0.5)

    def test_ap_empty_relevant(self):
        assert average_precision(["a", "b"], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests compute_all (integración)
# ---------------------------------------------------------------------------

class TestComputeAll:

    def test_returns_all_metric_keys(self):
        result = compute_all(["a", "b", "c"], ["a"], [], ks=[5, 10, 20])
        expected_keys = {"recall@5", "recall@10", "recall@20",
                         "precision@5", "precision@10", "precision@20",
                         "mrr", "ndcg@10", "ap"}
        assert set(result.keys()) == expected_keys

    def test_values_in_range(self, perfect_ranking):
        retrieved, relevant = perfect_ranking
        result = compute_all(retrieved, relevant, ["a"], ks=[5, 10])
        for v in result.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Tests load_gold_set
# ---------------------------------------------------------------------------

class TestLoadGoldSet:

    def _write_gold(self, data: dict, tmp_path) -> str:
        p = os.path.join(str(tmp_path), "gold.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return p

    def test_valid_gold_set_loads(self, tmp_path):
        data = {
            "version": "1.0",
            "queries": [
                {
                    "id": "q001",
                    "query": "Test query",
                    "preferences": {},
                    "relevant_poi_ids": ["poi_001"],
                    "highly_relevant_poi_ids": ["poi_001"],
                    "notes": "test",
                }
            ],
        }
        path = self._write_gold(data, tmp_path)
        queries = load_gold_set(path)
        assert len(queries) == 1
        assert queries[0]["id"] == "q001"

    def test_missing_queries_key_raises(self, tmp_path):
        path = self._write_gold({"version": "1.0"}, tmp_path)
        with pytest.raises(ValueError, match="queries"):
            load_gold_set(path)

    def test_missing_required_field_raises(self, tmp_path):
        data = {
            "queries": [
                {
                    "id": "q001",
                    # falta 'query', 'relevant_poi_ids', 'highly_relevant_poi_ids'
                }
            ]
        }
        path = self._write_gold(data, tmp_path)
        with pytest.raises(ValueError):
            load_gold_set(path)

    def test_real_gold_set_loads(self):
        gold_path = os.path.join(
            os.path.dirname(__file__), "..", "evaluation", "gold_set.json"
        )
        if not os.path.exists(gold_path):
            pytest.skip("gold_set.json no encontrado")
        queries = load_gold_set(gold_path)
        assert len(queries) == 40
        for q in queries:
            assert q["id"].startswith("q")
            assert isinstance(q["relevant_poi_ids"], list)
            assert isinstance(q["highly_relevant_poi_ids"], list)
