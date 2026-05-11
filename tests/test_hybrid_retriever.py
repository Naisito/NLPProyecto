import pytest

from app.hybrid_retriever import _rrf_fusion, _linear_fusion
from app.infra.bm25_index import BM25Index
from app.models import POI, Coordinates


def _make_poi(poi_id: str, name: str = None) -> POI:
    return POI(
        id=poi_id,
        name=name or poi_id,
        municipality="Bilbao",
        category="museo",
        subcategory="arte",
        description="desc",
        coordinates=Coordinates(lat=43.26, lon=-2.93),
        address="Calle 1",
        price="€",
        price_numeric=10.0,
        schedule={},
        source="test",
        url="",
        tags=[],
        enriched_text="texto de prueba",
        visit_duration_minutes=60,
        accessibility=True,
    )


class TestRRFFusion:
    def test_poi_in_both_rankings_gets_combined_score(self):
        poi_a = _make_poi("a")
        poi_b = _make_poi("b")

        dense = [(poi_a, 0.9), (poi_b, 0.7)]
        bm25 = [(poi_a, 0.5), (poi_b, 0.3)]

        fused = _rrf_fusion(dense, bm25, rrf_k=60)

        # Ambos están en ambos rankings → combinados
        assert len(fused) == 2

        # poi_a: rank 1 en dense → 1/61 + rank 1 en bm25 → 1/61 = 0.0328
        # poi_b: rank 2 en dense → 1/62 + rank 2 en bm25 → 1/62 = 0.0323
        assert fused[0][0].id == "a"
        assert fused[1][0].id == "b"

    def test_poi_only_in_one_ranking(self):
        poi_a = _make_poi("a")
        poi_b = _make_poi("b")
        poi_c = _make_poi("c")

        dense = [(poi_a, 0.9), (poi_b, 0.7)]
        bm25 = [(poi_c, 0.8), (poi_b, 0.3)]

        fused = _rrf_fusion(dense, bm25, rrf_k=60)

        # poi_b está en ambos → score combinado
        # poi_a solo en dense → score = 1/61
        # poi_c solo en bm25 → score = 1/61
        ids = [p[0].id for p in fused]
        assert "b" in ids  # combinado, probablemente primero
        assert "a" in ids
        assert "c" in ids
        assert len(fused) == 3

    def test_empty_dense_returns_bm25_only(self):
        poi_a = _make_poi("a")
        poi_b = _make_poi("b")

        dense = []
        bm25 = [(poi_a, 0.9), (poi_b, 0.7)]

        fused = _rrf_fusion(dense, bm25, rrf_k=60)
        assert len(fused) == 2
        assert fused[0][0].id == "a"
        assert fused[1][0].id == "b"

    def test_empty_both_returns_empty(self):
        fused = _rrf_fusion([], [], rrf_k=60)
        assert fused == []


class TestLinearFusion:
    def test_equal_weight(self):
        poi_a = _make_poi("a")
        poi_b = _make_poi("b")

        dense = [(poi_a, 1.0), (poi_b, 0.5)]
        bm25 = [(poi_a, 0.5), (poi_b, 1.0)]

        fused = _linear_fusion(dense, bm25, alpha=0.5)

        assert len(fused) == 2
        # Ambos deberían tener ~0.75 (normalizado)
        assert abs(fused[0][1] - fused[1][1]) < 0.01

    def test_dense_weighted_heavily(self):
        poi_a = _make_poi("a")
        poi_b = _make_poi("b")

        dense = [(poi_a, 1.0), (poi_b, 0.0)]
        bm25 = [(poi_a, 0.0), (poi_b, 1.0)]

        fused = _linear_fusion(dense, bm25, alpha=0.9)
        # poi_a dominates: 0.9*1.0 + 0.1*0.0 = 0.9
        # poi_b: 0.9*0.0 + 0.1*1.0 = 0.1
        assert fused[0][0].id == "a"
        assert fused[1][0].id == "b"

    def test_empty_inputs(self):
        fused = _linear_fusion([], [], alpha=0.5)
        assert fused == []
