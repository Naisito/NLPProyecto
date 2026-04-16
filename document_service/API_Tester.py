import os
import requests
import tkinter as tk
from tkinter import simpledialog, filedialog, scrolledtext, messagebox

BASE_URL = "http://localhost:8000"

# ----- Función para mostrar resultados genericos en una ventana nueva -----
def show_result(title, response):
    win = tk.Toplevel(root)
    win.title(title)

    text_area = scrolledtext.ScrolledText(win, width=100, height=40)
    text_area.pack(padx=10, pady=10)

    text_area.insert(tk.END, f"=== {title} ===\n")

    if isinstance(response, Exception):
        text_area.insert(tk.END, f"Error: {response}\n")
        return

    text_area.insert(tk.END, f"Status: {response.status_code}\n")

    try:
        text_area.insert(tk.END, f"Response: {response.json()}\n")
    except Exception:
        text_area.insert(tk.END, f"Response: {response.text}\n")

# ----- Función para mostrar resultados de busqueda (RAG) -----
def show_query_view(doc_id,llm_response, query, response):
    win = tk.Toplevel(root)
    win.title(f"Resultados: {query}")

    text_area = scrolledtext.ScrolledText(win, width=100, height=40)
    text_area.pack(padx=10, pady=10)

    if isinstance(response, Exception):
        text_area.insert(tk.END, f"Error consultando documento {doc_id}:\n{response}")
        return

    text_area.insert(tk.END, f"=== BuSQUEDA EN DOC {doc_id} ===\n")
    text_area.insert(tk.END, f"Pregunta: {query}\n")
    text_area.insert(tk.END, f"Status: {response.status_code}\n\n")

    if response.status_code != 200:
        text_area.insert(tk.END, f"Error: {response.text}")
        return

    try:
        data = response.json()
        results = data.get("results", [])
    except Exception:
        text_area.insert(tk.END, "Error parseando JSON.\n")
        return

    if llm_response:
        text_area.insert(tk.END, f"=== Respuesta LLM ===\n\n")
        llm_answer = data.get("llm_answer", "")
        text_area.insert(tk.END, llm_answer + "\n\n")

    text_area.insert(tk.END, f"=== Resultados de la búsqueda ===\n\n")    


    text_area.insert(tk.END, f"Coincidencias encontradas: {len(results)}\n")
    text_area.insert(tk.END, "-" * 80 + "\n\n")

    for i, chunk in enumerate(results, 1):
        text_area.insert(tk.END, f"--- Fragmento relevante #{i} ---\n")
        text_area.insert(tk.END, chunk.strip() + "\n\n")
        text_area.insert(tk.END, "-" * 40 + "\n\n")


# ----- Función especifica para mostrar un documento con su texto -----
def show_document_view(doc_id, response):
    win = tk.Toplevel(root)
    win.title(f"Documento {doc_id}")

    text_area = scrolledtext.ScrolledText(win, width=100, height=40)
    text_area.pack(padx=10, pady=10)

    if isinstance(response, Exception):
        text_area.insert(tk.END, f"Error obteniendo documento {doc_id}:\n{response}")
        return

    text_area.insert(tk.END, f"=== GET /documents/{doc_id} ===\n")
    text_area.insert(tk.END, f"Status: {response.status_code}\n\n")

    try:
        data = response.json()
    except Exception:
        text_area.insert(tk.END, "No se pudo parsear JSON.\n\n")
        text_area.insert(tk.END, f"Respuesta cruda:\n{response.text}")
        return

    if response.status_code != 200:
        text_area.insert(tk.END, "Error en la API:\n")
        text_area.insert(tk.END, f"{data}\n")
        return

    # Esperamos estructura: document_id, filename, full_text, created_at
    doc_id = data.get("document_id")
    filename = data.get("filename")
    created_at = data.get("created_at")
    short_summary = data.get("summary_short")
    long_summary = data.get("summary_long")
    full_text = data.get("full_text", "")

    text_area.insert(tk.END, "=== Metadatos ===\n")
    text_area.insert(tk.END, f"document_id: {doc_id}\n")
    text_area.insert(tk.END, f"filename: {filename}\n")
    text_area.insert(tk.END, f"created_at: {created_at}\n")
    text_area.insert(tk.END, f"len(full_text): {len(full_text)} caracteres\n\n")

    text_area.insert(tk.END, "=== Resumen Corto ===\n\n")
    text_area.insert(tk.END, short_summary + "\n\n")    
    text_area.insert(tk.END, "=== Resumen Largo ===\n\n")
    text_area.insert(tk.END, long_summary + "\n\n")

    text_area.insert(tk.END, "=== Contenido ===\n\n")
    text_area.insert(tk.END, full_text)

def show_summaries(title, response):
    win = tk.Toplevel(root)
    win.title(title)

    text_area = scrolledtext.ScrolledText(win, width=100, height=40)
    text_area.pack(padx=10, pady=10)

    text_area.insert(tk.END, f"=== {title} ===\n")

    if isinstance(response, Exception):
        text_area.insert(tk.END, f"Error: {response}\n")
        return

    text_area.insert(tk.END, f"Status: {response.status_code}\n\n")

    try:
        data = response.json()
    except Exception:
        text_area.insert(tk.END, "No se pudo parsear JSON.\n\n")
        text_area.insert(tk.END, f"Respuesta cruda:\n{response.text}")
        return

    if response.status_code != 200:
        text_area.insert(tk.END, "Error en la API:\n")
        text_area.insert(tk.END, f"{data}\n")
        return

    summary_short = data.get("summary_short", "")
    summary_long = data.get("summary_long", "")

    text_area.insert(tk.END, "=== Resumen Corto ===\n\n")
    text_area.insert(tk.END, summary_short + "\n\n")    
    text_area.insert(tk.END, "=== Resumen Largo ===\n\n")
    text_area.insert(tk.END, summary_long + "\n\n")

# ----- Botón GET /documents -----
def list_docs():
    try:
        r = requests.get(f"{BASE_URL}/documents")
        show_result("GET /documents", r)
    except Exception as e:
        show_result("GET /documents", e)


# ----- Botón POST /documents (upload) -----
def upload_document():
    file_path = filedialog.askopenfilename(
        title="Selecciona un PDF o TXT",
        filetypes=[("PDF/TXT/DOCX/Images Files", "*.pdf *.txt *.docx *.png *.jpg *.jpeg")],
    )

    if not file_path:
        return

    try:
        # Usar solo el nombre base para no meter la ruta completa
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"file": (filename, f)}
            r = requests.post(f"{BASE_URL}/documents", files=files)
        show_result("POST /documents", r)
    except Exception as e:
        show_result("POST /documents", e)


# ----- Botón GET /documents/{id} -----
def get_document():
    doc_id = simpledialog.askstring(
        "Documento", "Introduce el document_id:"
    )

    if not doc_id:
        return

    try:
        r = requests.get(f"{BASE_URL}/documents/{doc_id}")
        show_document_view(doc_id, r)
    except Exception as e:
        show_document_view(doc_id, e)


# ------ Boton GET /documents/{id}/query -----
def get_summaries():
    doc_id = simpledialog.askstring(
        "Resúmenes", "Introduce el document_id:"
    )

    if not doc_id:
        return

    try:
        r = requests.get(f"{BASE_URL}/documents/{doc_id}/summary")
        show_summaries(f"GET /documents/{doc_id}/summary", r)
    except Exception as e:
        show_summaries(f"GET /documents/{doc_id}/summary", e)


# ----- NUEVO: Botón POST /documents/{id}/query -----
def query_document():
    doc_id = simpledialog.askstring(
        "Busqueda Inteligente", "Introduce el document_id donde buscar:"
    )

    llm_answer = messagebox.askyesno(
        "Respuesta LLM",
        "¿Quieres que el LLM genere una respuesta basada en los fragmentos encontrados?"
    )

    query = simpledialog.askstring(
        "Busqueda Inteligente", "¿Que quieres saber del documento?"
    )
    if not query:
        return

    try:
        payload = {
            "query": query,
            "llm_answer": llm_answer 
        }
        
        r = requests.post(f"{BASE_URL}/documents/{doc_id}/query", json=payload)
        show_query_view(doc_id, llm_answer, query, r)
    except Exception as e:
        show_result(f"Error Query", e)


# ----- Botón DELETE /documents/{id} -----
def delete_document():
    doc_id = simpledialog.askstring(
        "Eliminar documento", "Introduce el document_id a eliminar:"
    )

    if not doc_id:
        return

    if not messagebox.askyesno(
        "Confirmar borrado",
        f"¿Seguro que quieres borrar el documento {doc_id}?"
    ):
        return

    try:
        r = requests.delete(f"{BASE_URL}/documents/{doc_id}")
        show_result(f"DELETE /documents/{doc_id}", r)
    except Exception as e:
        show_result(f"DELETE /documents/{doc_id}", e)


# ----- Botón DELETE /clear (vaciar BD) -----
def clear_database():
    if not messagebox.askyesno(
        "Vaciar base de datos",
        "⚠ Esto borrara TODOS los documentos y sus vectores.\n\n¿Estas seguro?"
    ):
        return

    try:
        r = requests.delete(f"{BASE_URL}/clear")
        show_result("DELETE /clear", r)
    except Exception as e:
        show_result("DELETE /clear", e)


# ----- Botón GET /health -----
def health():
    try:
        r = requests.get(f"{BASE_URL}/health")
        show_result("GET /health", r)
    except Exception as e:
        show_result("GET /health", e)


# ----- UI PRINCIPAL -----
root = tk.Tk()
root.title("Tester API Document Service + RAG")

title = tk.Label(root, text="Tester API - Document Service", font=("Arial", 18))
title.pack(pady=20)

# Botones organizados
frame = tk.Frame(root)
frame.pack(pady=10)

btn1 = tk.Button(frame, text="1. Listar Documentos", width=30, font=("Arial", 12), command=list_docs)
btn1.pack(pady=5)

btn2 = tk.Button(frame, text="2. Subir Documento (Upload)", width=30, font=("Arial", 12), command=upload_document)
btn2.pack(pady=5)

btn3 = tk.Button(frame, text="3. Ver Texto Completo (ID)", width=30, font=("Arial", 12), command=get_document)
btn3.pack(pady=5)

btnRsmn = tk.Button(frame, text="4. Ver Resúmenes (ID)", width=30, font=("Arial", 12), command=get_summaries)
btnRsmn.pack(pady=5)

# Nuevo botón destacado
btn_rag = tk.Button(frame, text="5. 🔍 Busqueda Semantica (RAG)", width=30, font=("Arial", 12, "bold"), bg="#e1f5fe", command=query_document)
btn_rag.pack(pady=15)

btn4 = tk.Button(frame, text="6. Borrar Documento", width=30, font=("Arial", 12), command=delete_document)
btn4.pack(pady=5)

btn5 = tk.Button(frame, text="7. Vaciar Base de Datos", width=30, font=("Arial", 12), fg="red", command=clear_database)
btn5.pack(pady=5)

btn6 = tk.Button(frame, text="Health Check", width=30, font=("Arial", 12), command=health)
btn6.pack(pady=5)

root.geometry("450x550")
root.mainloop()