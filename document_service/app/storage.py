import os
import sqlite3
from contextlib import contextmanager
from app import utils  
from datetime import datetime
from app.config import settings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Leer rutas desde config.json con fallbacks actuales
_upload_dir_cfg = settings.storage.get("upload_dir", "data")
_db_path_cfg = settings.storage.get("db_path", os.path.join("db", "documents.db"))

# Normalizar a rutas absolutas relativas al proyecto si son relativas
DATA_DIR = _upload_dir_cfg if os.path.isabs(_upload_dir_cfg) else os.path.join(BASE_DIR, _upload_dir_cfg)
DB_PATH = _db_path_cfg if os.path.isabs(_db_path_cfg) else os.path.join(BASE_DIR, _db_path_cfg)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

@contextmanager
def get_db_connection():
    """Context manager para asegurar que la conexión se cierra siempre."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT UNIQUE, 
            text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            summary_short TEXT NOT NULL,
            summary_long TEXT NOT NULL
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)")
        
        # Tabla para relación N:M entre sesiones y documentos
        cur.execute("""
        CREATE TABLE IF NOT EXISTS session_documents (
            session_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            added_at TEXT NOT NULL,
            PRIMARY KEY (session_id, document_id),
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_session_docs ON session_documents(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_sessions ON session_documents(document_id)")
        
        conn.commit()

def save_file_to_disk(filename: str, file_bytes: bytes) -> str:
    safe_name = filename.replace(" ", "_")
    timestamp = int(datetime.now().timestamp())
    path = os.path.join(DATA_DIR, f"{timestamp}_{safe_name}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

def save_document_record(document_id: str, filename: str, file_path: str, file_hash: str, text: str, summary_short: str , summary_long: str):
    created_at = utils.now_iso()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO documents(document_id, filename, file_path, file_hash, text, created_at, summary_short, summary_long) 
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (document_id, filename, file_path, file_hash, text, created_at, summary_short, summary_long)
        )
        conn.commit()
    return created_at

def get_document(document_id: str):
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
        row = cur.fetchone()
        if row:
            return dict(row)
    return None

def list_documents():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT document_id FROM documents")
        return [row["document_id"] for row in cur.fetchall()]

def find_document_by_hash(file_hash: str):
    """Busca duplicados instantáneamente por hash."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
        row = cur.fetchone()
        if row:
            return dict(row)
    return None

def delete_document(document_id: str):
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_path FROM documents WHERE document_id = ?", (document_id,))
        row = cur.fetchone()
        
        if not row:
            return False
        
        file_path = row["file_path"]
        cur.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        conn.commit()

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass
    return True

def clear_all():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_path FROM documents")
        paths = [row["file_path"] for row in cur.fetchall()]
        cur.execute("DELETE FROM session_documents")
        cur.execute("DELETE FROM documents")
        conn.commit()

    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def link_document_to_session(session_id: str, document_id: str):
    """Vincula un documento a una sesión."""
    created_at = utils.now_iso()
    with get_db_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO session_documents(session_id, document_id, added_at)
                VALUES (?, ?, ?)
                """,
                (session_id, document_id, created_at)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Ya estaba vinculado
            return False


def get_documents_by_session(session_id: str):
    """Obtiene todos los documentos vinculados a una sesión."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT d.document_id, d.filename, d.summary_short, d.summary_long
            FROM documents d
            JOIN session_documents sd ON d.document_id = sd.document_id
            WHERE sd.session_id = ?
            ORDER BY sd.added_at
        """, (session_id,))
        return [dict(row) for row in cur.fetchall()]


def unlink_document_from_session(session_id: str, document_id: str):
    """Desvincula un documento de una sesión."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM session_documents
            WHERE session_id = ? AND document_id = ?
        """, (session_id, document_id))
        conn.commit()
        return cur.rowcount > 0