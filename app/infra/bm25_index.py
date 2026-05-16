import logging
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

from nltk.stem import SnowballStemmer

logger = logging.getLogger("turismo_rag")

_SPANISH_STEMMER = SnowballStemmer("spanish")

# ~150 stopwords castellanas
_SPANISH_STOPWORDS: frozenset = frozenset({
    "a", "actualmente", "adelante", "además", "afirmó", "agregó", "ahí", "ahora",
    "al", "algún", "algo", "alguna", "algunas", "alguno", "algunos", "allá", "allí",
    "ambos", "ante", "antes", "aquel", "aquella", "aquellas", "aquello", "aquellos",
    "aquí", "arriba", "así", "aseguró", "aún", "aunque", "ayer",
    "bajo", "bastante", "bien", "breve",
    "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco",
    "comentó", "como", "con", "conocer", "considera", "consideró", "consigo",
    "contra", "cosas", "creo", "cual", "cuales", "cualquier", "cuando", "cuanto",
    "cuatro", "cuenta",
    "da", "dado", "dan", "dar", "de", "debe", "deben", "debido", "decir", "dejó",
    "del", "demás", "dentro", "desde", "después", "dice", "dicen", "dicho",
    "dieron", "diferente", "diferentes", "dijeron", "dijo", "dio", "donde",
    "dos", "durante",
    "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos", "embargo", "en",
    "encuentra", "entonces", "entre", "era", "eran", "es", "esa", "esas",
    "ese", "eso", "esos", "esta", "estaba", "estaban", "estado", "estar",
    "estará", "estas", "este", "esto", "estos", "estoy", "estuvo", "ex",
    "existe", "explicó",
    "fin", "fue", "fueron",
    "gran", "grandes",
    "ha", "haber", "había", "habían", "hace", "hacen", "hacer", "hacia",
    "han", "has", "hasta", "hay", "haya", "he", "hecho", "hemos", "hicieron",
    "hizo", "hombre", "hoy", "hubo",
    "igual", "indicó", "informó",
    "junto",
    "la", "lado", "las", "le", "les", "llegó", "lleva", "llevar", "lo",
    "los", "luego", "lugar",
    "manera", "manifestó", "mas", "mayor", "me", "mediante", "mejor",
    "mencionó", "menos", "mi", "mientras", "misma", "mismas", "mismo",
    "mismos", "modo", "mucha", "muchas", "mucho", "muchos", "muy",
    "nada", "nadie", "ningún", "ninguna", "no", "nos", "nosotras",
    "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "nueva",
    "nuevas", "nuevo", "nuevos", "nunca",
    "o", "ocho", "otra", "otras", "otro", "otros",
    "país", "para", "parece", "parte", "pasada", "pasado", "pero",
    "personas", "poca", "pocas", "poco", "pocos", "podemos", "podrá",
    "podrán", "podría", "podrían", "poner", "por", "porque", "posible",
    "primer", "primera", "primero", "principalmente", "propia", "propias",
    "propio", "propios", "pudo", "pueda", "puede", "pueden", "pues", "punto",
    "que", "quedó", "queremos", "quién", "quien",
    "realizó", "realizado", "respecto",
    "sabe", "se", "seis", "según", "ser", "será", "sería", "si",
    "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sin",
    "sino", "sobre", "sola", "solas", "solo", "solos", "son", "su", "sus",
    "tal", "también", "tampoco", "tan", "tanto", "tenemos", "tenía",
    "tendrá", "tendrán", "tener", "tercer", "ti", "tiene", "tienen",
    "toda", "todas", "todo", "todos", "total", "trabajo", "tras",
    "tres", "tuvo",
    "un", "una", "unas", "uno", "unos", "usted", "ustedes",
    "va", "vamos", "van", "varias", "varios", "veces", "ver", "vez",
    "y", "ya", "yo",
})


def _tokenize(text: str) -> List[str]:
    """Tokenizador español con stemming Snowball: lowercase + split + stem + stopwords."""
    tokens = text.lower().split()
    return [
        _SPANISH_STEMMER.stem(t)
        for t in tokens
        if t not in _SPANISH_STOPWORDS and len(t) > 1
    ]


class BM25Index:
    """Índice BM25 para retrieval lexical con tokenizador español (k1=1.5, b=0.75)."""

    def __init__(self, k1: float = None, b: float = None):
        from app.config import settings as _global_settings
        self.k1 = k1 if k1 is not None else _global_settings.bm25.get("k1", 1.5)
        self.b = b if b is not None else _global_settings.bm25.get("b", 0.75)

        self._documents: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._avg_dl: float = 0.0
        self._N: int = 0
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def build(self, documents: List[str]) -> None:
        self._documents = list(documents)
        self._N = len(documents)
        self._doc_tokens = []
        self._doc_len = []
        self._df = {}

        for doc in documents:
            tokens = _tokenize(doc)
            self._doc_tokens.append(tokens)
            self._doc_len.append(len(tokens))
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

        self._avg_dl = sum(self._doc_len) / max(self._N, 1)
        self._compute_idf()

        logger.info(
            f"BM25 index built: {self._N} docs, "
            f"avg_len={self._avg_dl:.1f}, vocab={len(self._df)}"
        )

    def _compute_idf(self) -> None:
        self._idf = {}
        for term, df in self._df.items():
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))
            self._idf[term] = max(idf, 0.0)

    # ------------------------------------------------------------------
    # Búsqueda
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        if self._N == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []
        for idx in range(self._N):
            score = self._score_doc(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        tokens = self._doc_tokens[doc_idx]
        doc_len = self._doc_len[doc_idx]
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for qt in query_tokens:
            idf = self._idf.get(qt, 0.0)
            if idf == 0.0:
                continue
            tf_q = tf.get(qt, 0)
            if tf_q == 0:
                continue
            num = tf_q * (self.k1 + 1)
            den = tf_q + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_dl, 1))
            score += idf * num / den

        return score

    def get_document(self, idx: int) -> Optional[str]:
        if 0 <= idx < self._N:
            return self._documents[idx]
        return None

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def persist(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "documents": self._documents,
            "doc_tokens": self._doc_tokens,
            "doc_len": self._doc_len,
            "avg_dl": self._avg_dl,
            "N": self._N,
            "df": self._df,
            "idf": self._idf,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 index persisted to {path}")

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.warning(f"BM25 index file not found: {path}")
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.k1 = data["k1"]
            self.b = data["b"]
            self._documents = data["documents"]
            self._doc_tokens = data["doc_tokens"]
            self._doc_len = data["doc_len"]
            self._avg_dl = data["avg_dl"]
            self._N = data["N"]
            self._df = data["df"]
            self._idf = data["idf"]
            logger.info(f"BM25 index loaded from {path}: {self._N} docs")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {path}: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._N > 0

    @property
    def num_docs(self) -> int:
        return self._N
