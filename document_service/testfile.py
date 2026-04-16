import pytest
import requests
import os
import time

# --- CONFIGURACIÓN ---
BASE_URL = os.getenv("API_URL", "http://localhost:8000")

print(f"\n⚡ Ejecutando tests contra: {BASE_URL}")

# --- FIXTURE: Setup/Teardown ---
@pytest.fixture(autouse=True)
def clean_db():
    """
    Limpia la base de datos REAL llamando al endpoint /clear antes de cada test.
    """
    try:
        response = requests.delete(f"{BASE_URL}/clear", timeout=10)
        if response.status_code != 200:
            pytest.fail(f"Error limpiando BD: {response.text}")
        
        # Esperamos para asegurar que ChromaDB y SQLite han liberado recursos
        time.sleep(1)
        
    except requests.exceptions.ConnectionError:
        pytest.fail(f"❌ No se pudo conectar a {BASE_URL}. ¿Está Docker corriendo?")
    
    yield

# --- TEST 0: Health Check (Nuevo) ---
def test_health_check():
    """
    Verifica que el servicio esté levantado y respondiendo.
    """
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# --- TEST 1: El "Happy Path" (Subida + RAG básico) ---
def test_upload_and_semantic_search():
    """
    Sube un documento y verifica que la búsqueda semántica funciona.
    """
    filename = "inteligencia.txt"
    content = b"La inteligencia artificial generativa permite crear contenido nuevo. ChatGPT es un ejemplo."
    
    files = {"file": (filename, content, "text/plain")}
    response = requests.post(f"{BASE_URL}/documents", files=files)
    
    assert response.status_code == 201, f"Error subiendo: {response.text}"
    doc_id = response.json()["document_id"]
    
    # Espera crítica para indexación vectorial en background
    time.sleep(1)
    
    # Búsqueda simple (sin LLM)
    query_payload = {"query": "crear textos", "llm_answer": False}
    response_query = requests.post(f"{BASE_URL}/documents/{doc_id}/query", json=query_payload)
    
    assert response_query.status_code == 200
    results = response_query.json()["results"]
    
    assert len(results) > 0
    assert "inteligencia artificial" in results[0]

# --- TEST 2: Resúmenes y Endpoint Específico (Nuevo) ---
def test_summary_generation_and_retrieval():
    """
    Verifica que se generen resúmenes (corto y largo) al subir,
    y que el endpoint específico de summaries funcione.
    """
    # Texto suficientemente largo para justificar un resumen
    text_content = (
        "Python es un lenguaje de programacion de alto nivel. "
        "Se utiliza para desarrollo web, ciencia de datos e inteligencia artificial. "
        "Su sintaxis es legible y limpia. " * 5
    )
    
    # 1. Subida
    files = {"file":("python_intro.txt", text_content.encode('utf-8'), "text/plain")}
    response = requests.post(f"{BASE_URL}/documents", files=files)
    assert response.status_code == 201
    
    data = response.json()
    doc_id = data["document_id"]
    
    # Verificar que la respuesta de creación ya trae resúmenes (según main.py líneas 124-125)
    assert "summary_short" in data
    assert "summary_long" in data
    # No verificamos contenido exacto porque depende del LLM, pero sí que no esté vacío
    assert len(data["summary_short"]) > 0 

    # 2. Probar el endpoint GET /documents/{id}/summary (Nuevo en main.py)
    summary_resp = requests.get(f"{BASE_URL}/documents/{doc_id}/summary")
    assert summary_resp.status_code == 200
    
    summary_data = summary_resp.json()
    assert "summary_short" in summary_data
    assert "summary_long" in summary_data
    assert summary_data["summary_short"] == data["summary_short"]

# --- TEST 3: Búsqueda con Respuesta LLM (Nuevo) ---
def test_query_with_llm_answer():
    """
    Prueba la generación de respuesta con LLM activando el flag llm_answer.
    """
    # Subimos un contexto claro
    content = b"El codigo secreto para entrar a la boveda es 1234-XYZ."
    requests.post(f"{BASE_URL}/documents", files={"file": ("secreto.txt", content, "text/plain")})
    
    # Esperamos indexación
    time.sleep(1)
    
    # Buscamos solicitando respuesta LLM
    # Nota: main.py espera 'llm_answer' en el body (DocumentQueryRequest)
    payload = {
        "query": "¿Cual es el codigo de la boveda?", 
        "llm_answer": True
    }
    
    # Hacemos la query global para variar (o específica si prefieres)
    response = requests.post(f"{BASE_URL}/documents//query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificamos que vienen resultados de vector
    assert len(data["results"]) > 0
    
    # Verificamos que viene la respuesta del LLM
    # Nota: Si no hay LLM configurado o falla, el main.py devuelve un string de error o vacío,
    # pero el campo debe existir en la respuesta.
    assert "llm_answer" in data
    assert isinstance(data["llm_answer"], str)
    # Opcional: imprimir para debug
    print(f"\n🤖 Respuesta LLM recibida: {data['llm_answer']}")

# --- TEST 4: Validación de Duplicados ---
def test_duplicate_upload_prevention():
    """
    Verifica que no se puedan subir dos archivos idénticos (mismo hash).
    """
    filename = "repetido.txt"
    content = b"Contenido unico para prueba de duplicados."
    
    # Primera subida
    files = {"file": (filename, content, "text/plain")}
    requests.post(f"{BASE_URL}/documents", files=files)
    time.sleep(1) 

    # Segunda subida (mismo contenido)
    files_again = {"file": (filename, content, "text/plain")}
    response2 = requests.post(f"{BASE_URL}/documents", files=files_again)
    
    assert response2.status_code == 409
    assert "Document already exists" in response2.json()["detail"]

def test_document_isolation():
    """
    Asegura que si busco en el Doc A, no me salen cosas del Doc B.
    """
    # 1. Subir Doc A
    resp_a = requests.post(f"{BASE_URL}/documents", files={"file": ("cocina.txt", b"Para hacer una paella necesitas arroz.", "text/plain")})
    assert resp_a.status_code == 201
    doc_a_id = resp_a.json()["document_id"]
    
    time.sleep(1)

    # 2. Subir Doc B
    resp_b = requests.post(f"{BASE_URL}/documents", files={"file": ("coches.txt", b"El motor V8 gasta gasolina.", "text/plain")})
    assert resp_b.status_code == 201
    doc_b_id = resp_b.json()["document_id"]

    time.sleep(3) # Damos un poco más de margen para indexar ambos

    # 3. Buscar "arroz" en el documento B (Coches)
    # --- CORRECCIÓN AQUÍ: Agregamos "llm_answer": False ---
    payload = {"query": "arroz", "llm_answer": False}
    
    response = requests.post(f"{BASE_URL}/documents/{doc_b_id}/query", json=payload)
    
    # Verificamos que no sea un error 422 antes de pedir el JSON
    assert response.status_code == 200, f"Error en la API: {response.text}"
    
    results = response.json()["results"]
    
    # Si devuelve resultados, asegurar que NO es el texto de la paella
    if results:
        assert "paella" not in results[0]
        
def test_global_search():
    """
    Prueba la búsqueda en TODOS los documentos.
    """
    requests.post(f"{BASE_URL}/documents", files={"file": ("doc1.txt", b"El sol es una estrella gigante.", "text/plain")})
    requests.post(f"{BASE_URL}/documents", files={"file": ("doc2.txt", b"La luna es un satelite natural.", "text/plain")})
    
    time.sleep(2) 

    # Usar /all/query Y añadir llm_answer: False
    response_global = requests.post(f"{BASE_URL}/documents//query", json={"query": "estrella", "llm_answer": False})
    
    assert response_global.status_code == 200, f"Error global: {response_global.text}"
    results = response_global.json()["results"]
    
    assert len(results) > 0
    assert "sol" in results[0]

def test_delete_document_and_vectors():
    content = b"Borrarme."
    resp = requests.post(f"{BASE_URL}/documents", files={"file": ("del.txt", content, "text/plain")})
    doc_id = resp.json()["document_id"]
    time.sleep(1)

    requests.delete(f"{BASE_URL}/documents/{doc_id}")
    time.sleep(1)

    get_response = requests.get(f"{BASE_URL}/documents/{doc_id}")
    assert get_response.status_code == 404