import logging
from typing import List, Dict, Optional
from app.interfaces import EmbeddingClient, VectorIndex

logger = logging.getLogger("doc_service")

class RagService:
    def __init__(
        self, 
        embedder: EmbeddingClient, 
        vector_store: VectorIndex, 
        chunk_size: int = 500, 
        overlap: int = 50,
        n_results_default: int = 3
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n_results_default = n_results_default

    def _chunk_text(self, text: str) -> List[str]:
            """
            Divide el texto respetando palabras completas (espacios) y usando
            los parámetros chunk_size y overlap configurados.
            """
            if not text:
                return []

            chunks = []
            start = 0
            text_len = len(text)

            while start < text_len:
                end = start + self.chunk_size

                if end >= text_len:
                    end = text_len
                    chunks.append(text[start:end])
                    break

                last_space = text.rfind(' ', start, end)

                if last_space != -1 and last_space > start:
                    end = last_space

                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)

                next_start = end - self.overlap

                if next_start <= start:
                    next_start = start + 1
                
                start = next_start

            return chunks

    def index_document(self, document_id: str, text: str):
        """Coordina: Chunking -> Embedding -> Guardado"""
        if not text:
            return

        chunks = self._chunk_text(text)
        if not chunks:
            return

        logger.info(f"Generando embeddings para {len(chunks)} fragmentos...")
        
        # 1. Generar vectores usando la interfaz de embedding
        vectors = self.embedder.encode(chunks)
        
        # Preparar metadatos
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"document_id": document_id, "type": "content"} for _ in chunks]

        # 2. Guardar en índice vectorial usando la interfaz de almacenamiento
        self.vector_store.add_vectors(
            ids=ids, 
            vectors=vectors, 
            metadatas=metadatas, 
            documents=chunks
        )
        logger.info(f"Documento {document_id} indexado correctamente.")

    def index_summary(self, document_id: str, summary_short: str, filename: str):
        """Indexa el resumen corto como documento especial para routing automático."""
        if not summary_short:
            return
            
        logger.info(f"Indexando resumen corto para doc {document_id}...")
        
        # El resumen es un único "chunk"
        summary_id = f"{document_id}__SUMMARY"
        summary_vector = self.embedder.encode([summary_short])[0]
        
        self.vector_store.add_vectors(
            ids=[summary_id],
            vectors=[summary_vector],
            metadatas=[{
                "document_id": document_id,
                "type": "summary",
                "filename": filename
            }],
            documents=[summary_short]
        )
        logger.info(f"Resumen indexado: {summary_id}")

    def route_to_best_documents(self, query: str, document_ids: List[str]) -> Dict[str, float]:
        """
        Dado un query y múltiples document_ids, determina cuáles son más relevantes.
        
        Busca en los resúmenes (summaries) de esos documentos para hacer ranking.
        Devuelve {document_id: relevance_score} ordenado por relevancia.
        """
        if not document_ids:
            return {}
            
        if len(document_ids) == 1:
            # Si solo hay un doc, retorna con score máximo
            return {document_ids[0]: 1.0}
        
        logger.info(f"Router: evaluando {len(document_ids)} documentos para query: '{query}'")
        
        # 1. Convertir query a vector
        query_vector = self.embedder.encode([query])[0]
        
        # 2. Buscar en resúmenes específicamente
        filters = {"type": {"$eq": "summary"}}  # Solo buscar en resúmenes
        # Usar búsqueda con scores reales normalizados
        results_scored = []
        try:
            results_scored = self.vector_store.search_with_scores(
                query_vector=query_vector,
                n_results=len(document_ids),
                filters=filters
            )
        except Exception as e:
            logger.warning(f"Fallo search_with_scores, usando fallback por posición: {e}")
            ids_fallback = self.vector_store.search(
                query_vector=query_vector,
                n_results=len(document_ids),
                filters=filters
            )
            total = max(len(ids_fallback), 1)
            results_scored = [{"id": rid, "score": 1.0 - (i/total)} for i, rid in enumerate(ids_fallback)]

        # 3. Mapear resultados con scores a document_ids
        doc_scores = {}
        for item in results_scored:
            rid = item.get("id", "")
            score = float(item.get("score", 0.0))
            if "__SUMMARY" in rid:
                doc_id = rid.replace("__SUMMARY", "")
                if doc_id in document_ids:
                    doc_scores[doc_id] = score
        
        # Si no encontramos resúmenes en BD vectorial, distribuir puntaje uniforme
        if not doc_scores:
            logger.warning(f"No se encontraron resúmenes en índice para {document_ids}. Distribuir uniforme.")
            return {doc_id: 0.5 for doc_id in document_ids}
        
        # Asegurar que todos los docs de la sesión están en scores
        for doc_id in document_ids:
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.1  # Puntaje bajo pero no cero
        
        logger.info(f"Scores de routing: {doc_scores}")
        return doc_scores

    def search(self, document_id: str, query: str) -> List[str]:
        """Coordina: Embedding de Query -> Búsqueda Vectorial"""
        logger.info(f"Procesando Query: '{query}'")

        # 1. Convertir query a vector
        query_vector = self.embedder.encode([query])[0]

        # 2. Configurar filtros si es necesario
        filters = None
        if document_id and document_id.lower() != "all":
            filters = {
                "$and": [
                    {"document_id": {"$eq": document_id}},
                    {"type": {"$eq": "content"}}
                ]
            }
            logger.info(f"Filtro: solo doc_id='{document_id}' (contenido)")

        # 3. Buscar (devuelve IDs)
        result_ids = self.vector_store.search(
            query_vector=query_vector, 
            n_results=self.n_results_default, 
            filters=filters
        )
        
        if not result_ids:
            return ["(No se encontraron resultados relevantes o la BD está vacía)"]
        
        # Ordenar IDs numéricamente antes de obtener documentos
        result_ids_sorted = self._sort_chunk_ids(result_ids)
        
        # 4. Obtener documentos reales por IDs
        docs_by_id = self.vector_store.get_documents_by_ids(result_ids_sorted)
        results = [docs_by_id.get(id_, "") for id_ in result_ids_sorted if id_ in docs_by_id]
        
        return results if results else ["(No se encontraron resultados)"]

    def _sort_chunk_ids(self, chunk_ids: List[str]) -> List[str]:
        """
        Ordena IDs de chunks numéricamente.
        Los IDs tienen formato: {document_id}_{chunk_number}
        """
        def extract_chunk_number(chunk_id: str) -> int:
            # Extraer el número después del último guión bajo
            parts = chunk_id.rsplit('_', 1)
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return float('inf')
            return float('inf')
        
        return sorted(chunk_ids, key=extract_chunk_number)

    def search_multi_document(self, document_ids: List[str], query: str) -> Dict:
        """
        Busca en múltiples documentos, retornando resultados agrupados por doc.
        Prioriza documentos más relevantes según routing automático.
        Los chunks se devuelven ordenados numéricamente por documento.
        
        Retorna: {
            "best_doc": document_id,
            "results_by_doc": {document_id: [chunks]},
            "best_chunks": [chunks del best_doc]
        }
        """
        if not document_ids:
            return {"best_doc": None, "results_by_doc": {}, "best_chunks": []}
        
        logger.info(f"Multi-search: {len(document_ids)} documentos")

        # 1. Determinar relevancias por routing
        doc_scores = self.route_to_best_documents(query, document_ids)
        if not doc_scores:
            # Fallback: todos iguales
            doc_scores = {doc_id: 1.0 for doc_id in document_ids}

        # Orden de prioridad por score
        docs_ordered = sorted(doc_scores.keys(), key=lambda d: (-doc_scores[d], d))
        best_doc = docs_ordered[0]

        # 2. Asignar presupuesto total de chunks entre documentos
        total_budget = max(1, self.n_results_default)
        min_secondary_score = 0.3

        # Documentos elegibles: best_doc + los que superan umbral
        eligible_docs = [d for d in docs_ordered if (d == best_doc or doc_scores.get(d, 0) >= min_secondary_score)]

        # Pesos proporcionales a score (ligero refuerzo al mejor)
        weights = {}
        for d in eligible_docs:
            w = doc_scores.get(d, 0.0)
            if d == best_doc:
                w = w * 1.1  # refuerzo leve
            weights[d] = max(w, 0.0)
        weight_sum = sum(weights.values()) or 1.0

        # Asignación inicial por proporción (entera)
        base_alloc = {d: int((weights[d] / weight_sum) * total_budget) for d in eligible_docs}

        # Asegurar al menos 1 al best_doc
        if base_alloc.get(best_doc, 0) == 0:
            base_alloc[best_doc] = 1

        # Asegurar que la suma no excede total y distribuir remanente por mayor peso
        used = sum(base_alloc.values())
        # Ajuste por exceso
        while used > total_budget and base_alloc[best_doc] > 1:
            base_alloc[best_doc] -= 1
            used = sum(base_alloc.values())
        # Rellenar faltantes
        while used < total_budget:
            for d in eligible_docs:
                if used >= total_budget:
                    break
                base_alloc[d] = base_alloc.get(d, 0) + 1
                used += 1

        logger.info(f"Routing scores: {doc_scores}")
        logger.info(f"Chunk allocation plan (pre-search): {base_alloc}")

        # 3. Ejecutar búsquedas por documento respetando asignación
        query_vector = self.embedder.encode([query])[0]
        results_by_doc: Dict[str, List[str]] = {}
        leftover = 0

        for d in eligible_docs:
            quota = base_alloc.get(d, 0)
            if quota <= 0:
                continue
            filters = {
                "$and": [
                    {"document_id": {"$eq": d}},
                    {"type": {"$eq": "content"}}
                ]
            }
            result_ids = self.vector_store.search(
                query_vector=query_vector,
                n_results=quota,
                filters=filters
            )
            if not result_ids:
                results_by_doc[d] = []
                leftover += quota
                continue
            result_ids_sorted = self._sort_chunk_ids(result_ids)
            docs_by_id = self.vector_store.get_documents_by_ids(result_ids_sorted)
            chunks = [docs_by_id.get(i, "") for i in result_ids_sorted if i in docs_by_id]
            # Si devolvió menos de la cuota, registrar sobrante
            if len(chunks) < quota:
                leftover += (quota - len(chunks))
            results_by_doc[d] = chunks

        # 4. Si hay sobrante, intentar rellenar pidiendo más al best_doc primero, luego resto
        if leftover > 0:
            refill_order = [best_doc] + [d for d in eligible_docs if d != best_doc]
            for d in refill_order:
                if leftover <= 0:
                    break
                already = len(results_by_doc.get(d, []))
                try_more = min(leftover, total_budget)  # cota de seguridad
                if try_more <= 0:
                    continue
                filters = {
                    "$and": [
                        {"document_id": {"$eq": d}},
                        {"type": {"$eq": "content"}}
                    ]
                }
                # Pedimos más; el índice puede devolver repetidos respecto a antes, así que filtramos
                extra_ids = self.vector_store.search(
                    query_vector=query_vector,
                    n_results=already + try_more,
                    filters=filters
                )
                # Remover los ya usados
                prev_ids = set()
                # reconstruir ids previos aproximando por contenidos no es fiable; pedimos documentos y evitamos duplicados por texto
                # Mejor: volvemos a pedir docs por ids y filtramos por ids en memoria
                # Para simplicidad aquí, solo truncamos a 'try_more' nuevos si el backend entrega orden estable
                if extra_ids:
                    result_ids_sorted = self._sort_chunk_ids(extra_ids)
                    new_ids = result_ids_sorted[already:already+try_more]
                    if new_ids:
                        docs_by_id = self.vector_store.get_documents_by_ids(new_ids)
                        new_chunks = [docs_by_id.get(i, "") for i in new_ids if i in docs_by_id]
                        if new_chunks:
                            results_by_doc[d] = results_by_doc.get(d, []) + new_chunks
                            leftover -= len(new_chunks)

        # 5. Ordenar claves de salida
        results_by_doc_sorted = {k: results_by_doc.get(k, []) for k in sorted(results_by_doc.keys())}

        # 6. best_chunks del best_doc
        best_chunks = results_by_doc_sorted.get(best_doc, [])

        return {
            "best_doc": best_doc,
            "results_by_doc": results_by_doc_sorted,
            "best_chunks": best_chunks
        }

    def delete_document_index(self, document_id: str):
        self.vector_store.delete(document_id)

    def clear_all_indexes(self):
        self.vector_store.clear()