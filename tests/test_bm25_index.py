import os
import tempfile

import pytest

from app.infra.bm25_index import BM25Index, _tokenize


class TestTokenize:
    def test_lowercase_and_split(self):
        tokens = _tokenize("Museo de Bellas Artes Bilbao")
        assert "museo" in tokens
        assert "bellas" in tokens
        assert "artes" in tokens

    def test_stopwords_removed(self):
        tokens = _tokenize("el museo de la ciudad")
        assert "el" not in tokens
        assert "de" not in tokens
        assert "la" not in tokens
        assert "museo" in tokens
        assert "ciudad" in tokens

    def test_short_tokens_removed(self):
        tokens = _tokenize("a b c de xy z")
        assert "xy" in tokens
        assert "a" not in tokens
        assert "z" not in tokens


class TestBM25IndexBuild:
    def test_build_empty(self):
        idx = BM25Index()
        idx.build([])
        assert idx.num_docs == 0
        assert not idx.is_loaded

    def test_build_from_documents(self):
        idx = BM25Index()
        docs = [
            "museo arte contemporáneo Bilbao",
            "playa surf deportes acuáticos Bizkaia",
            "museo historia patrimonio industrial",
        ]
        idx.build(docs)
        assert idx.num_docs == 3
        assert idx.is_loaded

    def test_search_by_exact_name(self):
        idx = BM25Index()
        docs = [
            "Museo Guggenheim arte contemporáneo Bilbao",
            "Playa de Ereaga Getxo arena mar",
            "Catedral de Santiago historia gótico Bilbao",
        ]
        idx.build(docs)
        results = idx.search("Guggenheim", k=2)
        assert len(results) > 0
        assert results[0][0] == 0  # primer documento

    def test_search_returns_relevant(self):
        idx = BM25Index()
        docs = [
            "parque natural Urkiola senderismo montaña",
            "museo Bellas Artes pintura escultura Bilbao",
            "playa Baquio surf arena mar Bizkaia",
        ]
        idx.build(docs)
        # Buscar playa → doc 2 debería tener el score más alto
        results = idx.search("playa surf mar", k=1)
        assert results[0][0] == 2

    def test_search_no_match(self):
        idx = BM25Index()
        docs = ["museo Bilbao", "playa Bizkaia"]
        idx.build(docs)
        results = idx.search("zzzxxxccc", k=5)
        assert results == []


class TestBM25IndexPersistence:
    def test_persist_and_load(self):
        idx = BM25Index()
        docs = [
            "museo Bilbao arte contemporáneo",
            "playa Getxo surf Bizkaia",
            "parque naturaleza senderismo",
        ]
        idx.build(docs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bm25.pkl")
            idx.persist(path)
            assert os.path.exists(path)

            idx2 = BM25Index()
            assert not idx2.is_loaded
            loaded = idx2.load(path)
            assert loaded
            assert idx2.is_loaded
            assert idx2.num_docs == 3

            # La búsqueda debe dar los mismos resultados
            r1 = idx.search("Guggenheim", k=3)
            r2 = idx2.search("Guggenheim", k=3)
            assert r1 == r2

    def test_load_nonexistent(self):
        idx = BM25Index()
        assert not idx.load("/nonexistent/path/bm25.pkl")
        assert not idx.is_loaded
