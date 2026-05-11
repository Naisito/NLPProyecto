import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("turismo_rag")

DB_PATH = Path("/app/db/routes.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                query      TEXT,
                score      REAL,
                exec_time  REAL,
                created_at TEXT NOT NULL,
                data       TEXT NOT NULL
            )
        """)
    logger.info("RouteStore inicializado en %s", DB_PATH)


def save_route(data: dict, query: str, exec_time: float) -> str:
    route_id   = str(uuid.uuid4())
    title      = data.get("route", {}).get("title", "Ruta sin título")
    score      = data.get("evaluation", {}).get("overall_score", 0.0)
    created_at = datetime.now().isoformat(timespec="seconds")
    with _conn() as c:
        c.execute(
            "INSERT INTO routes (id, title, query, score, exec_time, created_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (route_id, title, query or "", score, exec_time, created_at,
             json.dumps(data, ensure_ascii=False)),
        )
    logger.info("Ruta guardada: id=%s title=%r", route_id, title)
    return route_id


def list_routes(limit: int = 100) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, query, score, exec_time, created_at "
            "FROM routes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_route(route_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT data FROM routes WHERE id = ?", (route_id,)
        ).fetchone()
    return json.loads(row["data"]) if row else None


def delete_route(route_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM routes WHERE id = ?", (route_id,))
    return cur.rowcount > 0
