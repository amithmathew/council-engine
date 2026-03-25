"""SQLite database layer — schema, migrations, CRUD."""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path

from council.config import ATTACHMENTS_DIR, DB_DIR, DB_PATH


def get_db() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id                  INTEGER PRIMARY KEY,
            name                TEXT NOT NULL,
            session_id          TEXT,
            system_instruction  TEXT,
            created_at          TEXT DEFAULT (datetime('now')),
            updated_at          TEXT DEFAULT (datetime('now')),
            UNIQUE(name, session_id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            participant     TEXT NOT NULL,
            content         TEXT NOT NULL,
            attachments     TEXT,
            metadata        TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS attachments (
            id              TEXT PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            name            TEXT NOT NULL,
            mime_type       TEXT NOT NULL,
            size_bytes      INTEGER NOT NULL,
            file_path       TEXT NOT NULL,
            source_message  INTEGER,
            created_at      TEXT DEFAULT (datetime('now'))
        );
    """)
    # Migrations for existing DBs
    conv_cols = {r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    if "system_instruction" not in conv_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN system_instruction TEXT")
    msg_cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
    if "attachments" not in msg_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")
    if "metadata" not in msg_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
    conn.commit()
    conn.row_factory = sqlite3.Row
    return conn


def get_or_create_conversation(
    db: sqlite3.Connection,
    name: str,
    session_id: str | None,
    system_instruction: str | None = None,
) -> int:
    row = db.execute(
        "SELECT id FROM conversations WHERE name = ? AND session_id IS ?",
        (name, session_id),
    ).fetchone()
    if row:
        return row["id"]
    cur = db.execute(
        "INSERT INTO conversations (name, session_id, system_instruction) VALUES (?, ?, ?)",
        (name, session_id, system_instruction),
    )
    db.commit()
    return cur.lastrowid


def get_system_instruction(db: sqlite3.Connection, conversation_id: int) -> str | None:
    row = db.execute(
        "SELECT system_instruction FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    return row["system_instruction"] if row else None


def add_message(
    db: sqlite3.Connection,
    conversation_id: int,
    participant: str,
    content: str,
    attachments: list[dict] | None = None,
    metadata: dict | None = None,
) -> None:
    att_json = json.dumps(attachments) if attachments else None
    meta_json = json.dumps(metadata) if metadata else None
    db.execute(
        "INSERT INTO messages (conversation_id, participant, content, attachments, metadata) "
        "VALUES (?, ?, ?, ?, ?)",
        (conversation_id, participant, content, att_json, meta_json),
    )
    db.execute(
        "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
        (conversation_id,),
    )
    db.commit()


def get_messages(db: sqlite3.Connection, conversation_id: int) -> list[dict]:
    rows = db.execute(
        "SELECT participant, content, attachments, metadata, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    messages = []
    for r in rows:
        msg = dict(r)
        msg["attachments"] = json.loads(msg["attachments"]) if msg["attachments"] else None
        msg["metadata"] = json.loads(msg["metadata"]) if msg["metadata"] else None
        messages.append(msg)
    return messages


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

def store_attachment(
    db: sqlite3.Connection,
    conversation_id: int,
    source_path: str,
    name: str,
    mime_type: str,
) -> dict:
    """Copy a file to attachment storage and create a DB record. Returns attachment metadata."""
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    att_id = str(uuid.uuid4())
    src = Path(source_path)
    ext = src.suffix
    dest = ATTACHMENTS_DIR / f"{att_id}{ext}"
    shutil.copy2(str(src), str(dest))

    size_bytes = dest.stat().st_size
    db.execute(
        "INSERT INTO attachments (id, conversation_id, name, mime_type, size_bytes, file_path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (att_id, conversation_id, name, mime_type, size_bytes, str(dest)),
    )
    db.commit()
    return {
        "id": att_id,
        "name": name,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "file_path": str(dest),
    }


def list_attachments(db: sqlite3.Connection, conversation_id: int) -> list[dict]:
    """List all attachments for a conversation."""
    rows = db.execute(
        "SELECT id, name, mime_type, size_bytes, file_path, created_at "
        "FROM attachments WHERE conversation_id = ? ORDER BY created_at",
        (conversation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_attachment(db: sqlite3.Connection, attachment_id: str) -> dict | None:
    """Get a single attachment by ID."""
    row = db.execute(
        "SELECT id, name, mime_type, size_bytes, file_path, conversation_id, created_at "
        "FROM attachments WHERE id = ?",
        (attachment_id,),
    ).fetchone()
    return dict(row) if row else None
