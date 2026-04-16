import streamlit as st
import httpx
import time
import concurrent.futures
import os

# --- Configuración ---
AGENT_URL = os.getenv("AGENT_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="Document Agent Interface", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS (MODIFICADO) ---
st.markdown("""
    <style>
        /* 1. ALINEACIÓN VERTICAL */
        [data-testid="stHorizontalBlock"] {
            align-items: center; /* Centra verticalmente el texto y la papelera */
        }
        
        /* 2. BOTÓN DE BORRAR (TIPO PRIMARY -> ROJO COMPACTO) */
        button[kind="primary"] {
            background-color: #FF4B4B !important;
            border-color: #FF4B4B !important;
            color: white !important;
            border-radius: 6px;
            
            /* HACERLO PEQUEÑO */
            font-size: 12px !important;   /* Reduce el tamaño del emoji */
            height: auto !important;      /* Quita la altura forzada por defecto */
            min-height: 0px !important;   /* Permite que sea más bajito */
            padding: 4px 8px !important;  /* Relleno ajustado (Arriba/Abajo, Izq/Der) */
            line-height: 1.2 !important;
        }
        button[kind="primary"]:hover {
            background-color: #D93838 !important;
            border-color: #D93838 !important;
        }

        /* 3. BOTÓN DE SESIÓN (TIPO SECONDARY -> BLANQUECINO) */
        button[kind="secondary"] {
            text-align: left;
            background-color: transparent;
            border: 1px solid #4a4a4a;
            color: #e0e0e0;
            border-radius: 6px;
            height: auto !important;
            padding: 8px 10px !important; /* Un poco más de aire para el texto */
        }
        button[kind="secondary"]:hover {
            border-color: #ffffff;
            color: #ffffff;
            background-color: #262730;
        }
        button[kind="secondary"]:focus {
            border-color: #FF4B4B;
            color: white;
            background-color: #262730;
        }

        /* 4. BOTÓN NUEVA CONVERSACIÓN (EXCEPCIÓN VISUAL) */
        /* Como usamos 'primary' para borrar, el botón de arriba también se hará pequeño.
           Esto suele quedar bien (más elegante), pero si lo quieres grande avísame. */
        
        /* Ajuste de espacios */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.3rem; /* Menos espacio entre filas */
        }
    </style>
""", unsafe_allow_html=True)

# --- Estado ---
if "session_id" not in st.session_state: st.session_state.session_id = None
if "document_ids" not in st.session_state: st.session_state.document_ids = []
if "filenames" not in st.session_state: st.session_state.filenames = {}  # {document_id: filename}
if "messages" not in st.session_state: st.session_state.messages = [] 

# --- Funciones API ---

def get_all_sessions_api():
    try:
        resp = httpx.get(f"{AGENT_URL}/sessions", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"❌ No se puede conectar a {AGENT_URL}. ¿Los servicios están corriendo?")
        return []
    except Exception as e:
        st.error(f"Error listando sesiones: {e}")
        return []

def delete_session_api(session_id_to_delete):
    try:
        url = f"{AGENT_URL}/sessions/{session_id_to_delete}"
        resp = httpx.delete(url, timeout=5.0)
        resp.raise_for_status()
        
        if st.session_state.session_id == session_id_to_delete:
            st.session_state.session_id = None
            st.session_state.document_ids = []
            st.session_state.messages = []
            
        st.toast("Conversación eliminada.", icon="🗑️")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        st.error(f"Error borrando: {e}")

def create_session_api():
    try:
        with st.spinner("Creando..."):
            resp = httpx.post(f"{AGENT_URL}/sessions", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.document_ids = []
            st.session_state.filenames = {}
            st.session_state.messages = []
            st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

def load_session_api(session_id_input):
    if not session_id_input: return
    if st.session_state.session_id == session_id_input: return
    
    try:
        with st.spinner("Cargando..."):
            url = f"{AGENT_URL}/sessions/{session_id_input}"
            resp = httpx.get(url, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            
            st.session_state.session_id = data["session_id"]
            st.session_state.document_ids = data.get("document_ids", [])
            st.session_state.filenames = data.get("filenames", {})
            
            history_data = data.get("chat_history", [])
            restored = []
            for item in history_data:
                if "user" in item:
                    restored.append({"role": "user", "content": item["user"]})
                if "bot" in item:
                    # Intentar cargar chunks agrupados por documento
                    chunks_by_doc = item.get("results_by_document", {})
                    if not chunks_by_doc:
                        # Fallback: usar chunks simples
                        chunks = item.get("chunks", [])
                        if chunks:
                            chunks_by_doc = {"chunks": chunks}
                    
                    source_info = ""
                    if item.get("source_filename"):
                        source_info = f" (Desde: {item['source_filename']})"
                    restored.append({
                        "role": "assistant", 
                        "content": item["bot"] + source_info,
                        "sources": chunks_by_doc
                    })
            st.session_state.messages = restored
            st.rerun()
    except Exception as e:
        st.error(f"Error cargando: {e}")

def upload_document_api(uploaded_file):
    if not st.session_state.session_id: return
    try:
        with st.spinner("Subiendo..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            url = f"{AGENT_URL}/sessions/{st.session_state.session_id}/upload_document"
            resp = httpx.post(url, files=files, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            doc_id = data["document_id"]
            
            # Actualizar lista de documentos en sesión
            if doc_id not in st.session_state.document_ids:
                st.session_state.document_ids.append(doc_id)
            # Guardar el nombre del archivo
            st.session_state.filenames[doc_id] = uploaded_file.name
            
            st.success(f"Archivo vinculado. Total documentos: {data.get('total_docs', len(st.session_state.document_ids))}")
            time.sleep(0.5)
            st.rerun()
    except Exception as e:
         st.error(f"Error subida: {e}")

def upload_documents_api(uploaded_files):
    """Sube varios documentos en paralelo a la sesión actual."""
    if not st.session_state.session_id:
        return

    if not uploaded_files:
        return

    session_id = st.session_state.session_id

    def _post_file(u_file):
        files = {"file": (u_file.name, u_file.getvalue(), u_file.type)}
        url = f"{AGENT_URL}/sessions/{session_id}/upload_document"
        # Aumentamos timeout por posible concurrencia / archivos grandes
        resp = httpx.post(url, files=files, timeout=180.0)
        resp.raise_for_status()
        return resp.json()

    successes = 0
    added_ids = []
    errors = []
    total = len(uploaded_files)
    completed = 0

    progress_bar = st.progress(0)
    progress_text = st.empty()

    with st.spinner("Subiendo archivos en paralelo..."):
        # Limitamos workers para no saturar el backend
        max_workers = min(4, len(uploaded_files))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_post_file, f): f.name for f in uploaded_files}
            for future in concurrent.futures.as_completed(future_map):
                fname = future_map[future]
                try:
                    data = future.result()
                    doc_id = data.get("document_id")
                    if doc_id and doc_id not in st.session_state.document_ids:
                        st.session_state.document_ids.append(doc_id)
                        # Guardar el nombre del archivo
                        st.session_state.filenames[doc_id] = data.get("filename", f"Doc {doc_id[:8]}")
                        added_ids.append(doc_id)
                    successes += 1
                except Exception as e:
                    errors.append((fname, str(e)))
                finally:
                    completed += 1
                    percent = int((completed / total) * 100)
                    progress_bar.progress(percent)
                    progress_text.write(f"Subido {completed}/{total}: {fname}")

    if successes:
        st.success(f"{successes} archivo(s) subido(s). Documentos activos: {len(st.session_state.document_ids)}")
    if errors:
        for fname, msg in errors:
            st.error(f"Error con {fname}: {msg}")
    progress_bar.progress(100)
    progress_text.write("Carga completada.")
    time.sleep(0.5)
    st.rerun()

def send_chat_api(query):
    """Envía query con o sin documentos. Automáticamente elige el endpoint correcto."""
    sid = st.session_state.session_id
    try:
        # Si hay documentos, usa endpoint multi-documento
        if st.session_state.document_ids:
            url = f"{AGENT_URL}/sessions/{sid}/chat"
        # Si no hay documentos, usa endpoint de chat general
        else:
            url = f"{AGENT_URL}/sessions/{sid}/chat_general"
        
        resp = httpx.post(url, json={"query": query}, timeout=90.0)
        resp.raise_for_status()
        data = resp.json()
        
        # Extraer chunks agrupados por documento si está disponible
        chunks_by_doc = {}
        if "results_by_document" in data:
            chunks_by_doc = data.get("results_by_document", {})
        else:
            # Fallback: agrupar chunks simples por documento
            used_chunks = data.get("used_chunks", [])
            if used_chunks and not chunks_by_doc:
                chunks_by_doc = {"chunks": used_chunks}
        
        return data.get("answer", ""), chunks_by_doc
    except Exception as e:
        st.error(f"Error chat: {e}")
        return "Error.", {}

# --- BARRA LATERAL ---
with st.sidebar:
    # Botón "Nueva Conversación" (Se verá un poco más compacto también, queda elegante)
    if st.button("+ Nueva Conversación", use_container_width=True, type="primary"):
        create_session_api()
    
    st.divider()

    st.subheader("Conversaciones")
    sessions_list = get_all_sessions_api()
    
    if not sessions_list:
        st.caption("No hay historial.")
    else:
        for s in sessions_list:
            s_id = s['session_id']
            doc_names = s['filenames'] if s['filenames'] else "Sin documentos"
            doc_count = s.get('document_count', 0)
            
            # Recortar nombre
            if len(doc_names) > 20: 
                doc_names = doc_names[:17] + "..."
            
            label = f"{doc_names} ({s['messages_count']})"
            
            # Ajustamos columnas: Más espacio al nombre (0.82), menos a la papelera (0.18)
            col1, col2 = st.columns([0.82, 0.18])
            
            with col1:
                # Botón Cargar (Secondary -> Blanquecino)
                if st.button(label, key=f"load_{s_id}", use_container_width=True, type="secondary"):
                    load_session_api(s_id)
            
            with col2:
                # Botón Borrar (Primary -> Rojo Compacto)
                if st.button("🗑️", key=f"del_{s_id}", use_container_width=True, type="primary"):
                    delete_session_api(s_id)

    st.divider()
    
    if st.session_state.session_id:
        st.caption("**Agregar documentos (opcional):**")
        uploads = st.file_uploader(
            "Archivos",
            type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="upload_multi",
        )
        if uploads and st.button("Agregar archivo(s)"):
            # Si solo viene uno, usamos la función simple para compatibilidad
            if isinstance(uploads, list) and len(uploads) > 1:
                upload_documents_api(uploads)
            else:
                upload_document_api(uploads[0] if isinstance(uploads, list) else uploads)
        
        if st.session_state.document_ids:
             st.success(f"📄 {len(st.session_state.document_ids)} documentos activos")

# --- PANEL PRINCIPAL ---
st.title("U-NA.AI Agente de Documentos")

if not st.session_state.session_id:
    st.info("👈 Pulsa '+ Nueva Conversación' para empezar.")

# Render Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Referencias"):
                sources = msg["sources"]
                # Si es dict (agrupado por documento)
                if isinstance(sources, dict):
                    for doc_id in sorted(sources.keys()):
                        chunks = sources[doc_id]
                        # Obtener nombre del documento desde filenames
                        doc_name = st.session_state.filenames.get(doc_id, doc_id)
                        st.markdown(f"### 📄 {doc_name}")
                        for i, chunk in enumerate(chunks, 1):
                            st.text(f"[Chunk {i}]\n{chunk}")
                # Si es lista (formato antiguo)
                else:
                    for s in sources: 
                        st.text(s)

# Input Chat
if st.session_state.session_id:
    chat_placeholder = st.chat_input("Escribe tu pregunta..." if st.session_state.document_ids else "Inicia una conversación...")
    if chat_placeholder:
        prompt = chat_placeholder
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Pensando..."):
                ans, srcs = send_chat_api(prompt)
                
                step_size = 5
                for i in range(0, len(ans), step_size):
                    chunk = ans[i:i+step_size]
                    full_response += chunk
                    time.sleep(0.015)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(ans)

                if srcs:
                    with st.expander("Referencias"):
                        # Si srcs es dict (agrupado por documento)
                        if isinstance(srcs, dict):
                            for doc_id in sorted(srcs.keys()):
                                chunks = srcs[doc_id]
                                # Obtener nombre del documento desde filenames
                                doc_name = st.session_state.filenames.get(doc_id, doc_id)
                                st.markdown(f"### 📄 {doc_name}")
                                for i, chunk in enumerate(chunks, 1):
                                    st.text(f"[Chunk {i}]\n{chunk}")
                        # Si es lista (formato antiguo)
                        else:
                            for s in srcs: 
                                st.text(s)
        
        st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})