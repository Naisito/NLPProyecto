import os
import concurrent.futures
from openai import OpenAI
from app.config import settings
from typing import List, Union, Any, Dict, Optional

api_key = settings.llm.get("openai_api_key")
model = settings.llm.get("openai_model_name")

# Importar parámetros desde config (centralizados)
LLM_CONFIG = settings.llm
TEMPERATURE_MAP = LLM_CONFIG.get("temperature_map", 0.2)
TEMPERATURE_REDUCE = LLM_CONFIG.get("temperature_reduce", 0.3)
TEMPERATURE_ANSWER = LLM_CONFIG.get("temperature_answer", 0.1)
TEMPERATURE_CHAT = LLM_CONFIG.get("temperature_chat", 0.3)
MAX_TOKENS_SUMMARY = LLM_CONFIG.get("max_tokens_summary", 300)
KEEP_LAST_N_MESSAGES = LLM_CONFIG.get("keep_last_n_messages", 5)

client = OpenAI(api_key=api_key)

def short_summary(text: str) -> str:
    """
    Genera un resumen tipo 'Keywords/Tags' para el ROUTER o Búsqueda Vectorial.
    Objetivo: Densidad semántica máxima.
    """
    if not text:
        return ""
        
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un extractor de metadatos técnicos. Tu salida debe ser densa y directa."
                },
                {
                    "role": "user",
                    "content": (
                        f"Analiza el siguiente texto y genera un perfil de metadatos de MÁXIMO 40 palabras.\n"
                        f"OBJETIVO: Diferenciar este documento de millones de otros en una base de datos vectorial.\n"
                        f"REGLAS:\n"
                        f"1. NO uses frases como 'Este texto trata de...'. Ve al grano.\n"
                        f"2. PRIORIZA: Nombres propios, códigos de proyecto, fechas, ubicación geográfica, tema central y tipo de documento (Contrato, Novela, Manual).\n"
                        f"3. FORMATO: Texto plano separado por comas o puntos.\n\n"
                        f"Texto:\n{text[:15000]}" 
                    )
                }
            ], 
            temperature=TEMPERATURE_ANSWER, 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generando short_summary: {e}")
        return "Etiquetas no disponibles."

def map_summary_chunk(text: str) -> str:
    """
    MAP FUNCTION: Resume un fragmento detectando automáticamente su tipo (Narrativo, Técnico, Legal).
    """
    if not text: 
        return ""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en síntesis documental. Tu objetivo es comprimir información "
                        "preservando el tono, la estructura y los datos duros originales."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Analiza el siguiente fragmento y genera un resumen detallado.\n"
                        f"INSTRUCCIONES DE ADAPTACIÓN:\n"
                        f"1. SI ES NARRATIVA: Resume la trama en prosa fluida. Céntrate en personajes y eventos.\n"
                        f"2. SI ES TÉCNICO/MANUAL: Conserva pasos, comandos y especificaciones. Usa listas si ayuda a la claridad.\n"
                        f"3. SI ES LEGAL/FORMAL: Conserva cláusulas, definiciones y obligaciones. Mantén lenguaje formal.\n\n"
                        f"REGLA DE ORO: No pierdas nombres propios, fechas, cifras o términos técnicos clave.\n\n"
                        f"INSTRUCCIÓN NEGATIVA: Ignora estrictamente pies de página, notas de copyright, "
                        f"licencias de distribución (Project Gutenberg, Creative Commons) o mensajes editoriales modernos.\n\n"
                        f"Fragmento a resumir:\n{text}"
                    )
                }
            ],
            temperature=TEMPERATURE_MAP, 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error en map_summary_chunk: {e}")
        return f"[Error procesando fragmento]"

def reduce_summaries(summaries_text: str) -> str:
    """
    REDUCE FUNCTION: Toma una lista de resúmenes y crea un documento final cohesivo.
    Elimina las costuras entre fragmentos ("Parte 1", "Parte 2").
    """
    if not summaries_text: 
        return ""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un Editor Jefe experto en redacción y unificación de textos complejos."
                },
                {
                    "role": "user",
                    "content": (
                        f"A continuación tienes una serie de resúmenes secuenciales extraídos de un documento más grande.\n"
                        f"Tu tarea es UNIFICARLOS en un ÚNICO texto coherente y bien estructurado.\n\n"
                        f"REGLAS OBLIGATORIAS:\n"
                        f"1. UNIFICACIÓN: El resultado final NO debe parecer un collage. Elimina menciones a 'Fragmento', 'Parte 1', etc.\n"
                        f"2. ESTILO: Si el contenido es narrativo, usa conectores temporales. Si es técnico, mantén la lógica estructural.\n"
                        f"3. COMPLETITUD: Asegúrate de que el inicio y el final tengan sentido por sí solos.\n"
                        f"4. SÍNTESIS: Si hay redundancia en las uniones de los fragmentos, fusiónala.\n\n"
                        f"Resúmenes a procesar:\n{summaries_text}"
                    )
                }
            ],
            temperature=TEMPERATURE_REDUCE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error en reduce_summaries: {e}")
        return "Error generando resumen final."

def generate_answer(
    question: str, 
    summary_long: str, 
    context: Union[str, List[Any]], 
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Genera respuesta usando RAG (Retrieval-Augmented Generation).
    
    UNIFICACIÓN de:
    - answer_with_context() 
    - answer_with_context_and_chat()
    
    Args:
        question: Pregunta del usuario
        summary_long: Resumen del documento
        context: Fragmentos relevantes (string o lista) - puede estar vacío para chat general
        chat_history: Opcional. Historial de conversación previo
    
    Returns:
        Respuesta generada por el LLM
    """
    # Si no hay contexto, modo chat general sin documentos
    if not context and not summary_long:
        system_instruction = (
            "Eres un asistente útil y conversacional. "
            "Responde de manera amigable y precisa a las preguntas del usuario."
        )
    else:
        # Formateo de contexto cuando existe
        if isinstance(context, list):
            formatted_context = "\n".join([f"Fragmento {i+1}: {str(item)}" for i, item in enumerate(context)])
        else:
            formatted_context = str(context)

        # Construir sistema prompt con documentos
        system_instruction = (
            "Eres un asistente útil y preciso. Responde a la pregunta del usuario basándote en este orden de prioridad:\n"
            "1. FRAGMENTOS DEL DOCUMENTO (contexto abajo).\n"
            "2. HISTORIAL DE CONVERSACIÓN (para contexto de la charla).\n"
            "3. CONOCIMIENTO GENERAL (solo si la respuesta no está en el documento).\n\n"
            f"--- RESUMEN DEL DOCUMENTO ---\n{summary_long}\n\n"
            f"--- FRAGMENTOS RELEVANTES ---\n{formatted_context}\n"
        )

    # Construir mensajes
    messages = [{"role": "system", "content": system_instruction}]

    # Agregar historial de chat si existe
    if chat_history:
        for turn in chat_history:
            if turn.get("user"):
                messages.append({"role": "user", "content": turn["user"]})
            if turn.get("bot"):
                messages.append({"role": "assistant", "content": turn["bot"]})

    # Agregar pregunta actual
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE_CHAT if chat_history else TEMPERATURE_ANSWER,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error LLM: {e}")
        return "Error generando respuesta."

def smart_split(text: str, chunk_size: int, overlap: int = 200) -> List[str]:
    """
    Divide texto respetando palabras/espacios.
    Evita partir en mitad de una palabra.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            # Buscar el último espacio antes de 'end'
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        
        chunks.append(text[start:end])
        start = end - overlap
        if start >= end: 
            start = end

    return chunks

def summarize_with_map_reduce(
    text: str,
    chunk_size: int = 15000,
    overlap: int = 500,
    max_workers: int = 5
) -> str:
    """
    Algoritmo Map-Reduce optimizado para procesar textos grandes.
    
    UNIFICACIÓN de:
    - map_reduce_summary()
    
    Flow:
    1. Divide el texto en chunks (MAP)
    2. Resumen paralelo de cada chunk (MAP paralelo)
    3. Reduce recursivo si resultado es muy grande
    4. Reduce final (REDUCE)
    
    Args:
        text: Texto a resumir
        chunk_size: Tamaño de cada chunk (caracteres)
        overlap: Solapamiento entre chunks
        max_workers: Threads paralelos
    
    Returns:
        Resumen final del documento
    """
    if not text:
        return ""

    if len(text) <= chunk_size:
        # Texto pequeño: resumen directo
        return reduce_summaries(text)

    # MAP: Dividir en chunks y resumir cada uno en paralelo
    chunks = smart_split(text, chunk_size, overlap)
    print(f"--> [Map-Reduce] Iniciando: {len(chunks)} fragmentos de {chunk_size} chars.")

    intermediate_summaries = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(map_summary_chunk, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        results_map = {}
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                data = future.result()
                results_map[idx] = data
            except Exception as e:
                print(f"Error en chunk {idx}: {e}")
                results_map[idx] = ""

    # Ordenar por índice original
    for i in range(len(chunks)):
        if i in results_map and results_map[i]:
            intermediate_summaries.append(f"--- Fragmento {i+1} ---\n{results_map[i]}")

    combined_text = "\n\n".join(intermediate_summaries)
    
    # REDUCE: Verificar si necesita recursión
    SAFE_CONTEXT_LIMIT = 120000 

    if len(combined_text) > SAFE_CONTEXT_LIMIT:
        print(f"--> [Recursión] Texto combinado ({len(combined_text)} chars) excede límite ({SAFE_CONTEXT_LIMIT}). Iterando...")
        return summarize_with_map_reduce(
            combined_text, 
            chunk_size=chunk_size, 
            overlap=overlap,
            max_workers=max_workers
        )
    else:
        # Reduce final
        print(f"--> [Reduce Final] Texto combinado ({len(combined_text)} chars) cabe en contexto.")
        return reduce_summaries(combined_text)
    
def optimize_chat_history(chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optimiza el historial de chat para evitar exceso de tokens.
    
    UNIFICACIÓN de:
    - summarize_chat_history()
    
    Strategy:
    1. Mantener últimos N mensajes (contexto reciente)
    2. Resumir mensajes anteriores (contexto histórico comprimido)
    3. Devolver [RESUMEN] + [ÚLTIMOS N MENSAJES]
    
    Args:
        chat_history: Lista de turnos {user, bot}
    
    Returns:
        Historial optimizado
    """
    if not chat_history:
        return []

    # Si es corto, no resumir
    if len(chat_history) <= KEEP_LAST_N_MESSAGES:
        return chat_history

    older_messages = chat_history[:-KEEP_LAST_N_MESSAGES]
    recent_messages = chat_history[-KEEP_LAST_N_MESSAGES:]

    # Construir texto a resumir
    text_to_summarize = ""
    for turn in older_messages:
        u = turn.get("user", "")
        b = turn.get("bot", "")
        if u: 
            text_to_summarize += f"User: {u}\n"
        if b: 
            text_to_summarize += f"Bot: {b}\n"

    try:
        response = client.chat.completions.create(
            model=model, 
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Eres un experto resumiendo conversaciones. "
                        "Genera un resumen que capture hechos clave, "
                        "nombres y contexto."
                    )
                },
                {"role": "user", "content": text_to_summarize}
            ],
            temperature=TEMPERATURE_ANSWER,
            max_tokens=MAX_TOKENS_SUMMARY
        )
        summary_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error resumiendo chat: {e}")
        summary_text = "Resumen no disponible."

    # Crear turno de resumen
    summary_turn = {
        "user": f"[CONTEXTO ANTERIOR COMPRIMIDO]:\n{summary_text}",
        "bot": "Entendido. Usaré este contexto para las respuestas."
    }

    return [summary_turn] + recent_messages