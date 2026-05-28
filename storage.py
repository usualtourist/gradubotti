import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor


# ============================================================
# Ympäristöasetukset
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)


# ============================================================
# Yleiset apufunktiot
# ============================================================

def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def get_database_url() -> Optional[str]:
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return database_url

    try:
        import streamlit as st
        database_url = st.secrets.get("DATABASE_URL", None)
        if database_url:
            return database_url
    except Exception:
        pass

    return None


def get_connection():
    database_url = get_database_url()

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL puuttuu. Lisää se .env-tiedostoon tai Streamlitin secrets-asetuksiin. "
            "Tarkistettu .env-polku: " + str(ENV_PATH)
        )

    return psycopg2.connect(
        database_url,
        cursor_factory=RealDictCursor
    )


def safe_close(conn=None, cur=None):
    try:
        if cur:
            cur.close()
    except Exception:
        pass

    try:
        if conn:
            conn.close()
    except Exception:
        pass


# ============================================================
# Tietokannan alustus
# ============================================================

def init_db():
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id)
                    REFERENCES users(user_id)
                    ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                mode TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id)
                    REFERENCES users(user_id)
                    ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS course_materials (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id)
                    REFERENCES users(user_id)
                    ON DELETE CASCADE,
                UNIQUE(user_id, file_hash)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS checkins (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                completed TEXT,
                blocked TEXT,
                next_action TEXT,
                support_needed TEXT,
                coach_response TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id)
                    REFERENCES users(user_id)
                    ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_user_id_id
            ON messages(user_id, id)
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkins_user_id_id
            ON checkins(user_id, id)
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_course_materials_user_id_id
            ON course_materials(user_id, id)
            """
        )

        conn.commit()

    finally:
        safe_close(conn, cur)


# ============================================================
# Diagnostiikka ja terveystarkistus
# ============================================================

def database_health_check() -> Dict[str, Any]:
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT 1 AS ok")
        row = cur.fetchone()

        return {
            "ok": bool(row and row["ok"] == 1),
            "message": "Tietokantayhteys toimii"
        }

    except Exception as e:
        return {
            "ok": False,
            "message": "Tietokantayhteys epäonnistui: " + str(e)
        }

    finally:
        safe_close(conn, cur)


def debug_database_url_status() -> Dict[str, Any]:
    database_url = get_database_url()

    if database_url:
        preview = database_url[:25] + "..."
    else:
        preview = None

    return {
        "env_path": str(ENV_PATH),
        "env_exists": ENV_PATH.exists(),
        "database_url_found": bool(database_url),
        "database_url_preview": preview
    }


# ============================================================
# Käyttäjät
# ============================================================

def ensure_user(user_id: str):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO users (user_id, created_at)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO NOTHING
            """,
            (user_id, utc_now())
        )

        conn.commit()

    finally:
        safe_close(conn, cur)


# ============================================================
# Profiilit
# ============================================================

def save_profile(user_id: str, profile: dict):
    ensure_user(user_id)

    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO profiles (user_id, profile_json, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET
                profile_json = EXCLUDED.profile_json,
                updated_at = EXCLUDED.updated_at
            """,
            (
                user_id,
                json.dumps(profile, ensure_ascii=False),
                utc_now()
            )
        )

        conn.commit()

    finally:
        safe_close(conn, cur)


def load_profile(user_id: str) -> dict:
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT profile_json
            FROM profiles
            WHERE user_id = %s
            """,
            (user_id,)
        )

        row = cur.fetchone()

        if not row:
            return {}

        try:
            return json.loads(row["profile_json"])
        except Exception:
            return {}

    finally:
        safe_close(conn, cur)


# ============================================================
# Viestihistoria
# ============================================================

def save_message(user_id: str, role: str, content: str, mode: str = None):
    ensure_user(user_id)

    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO messages (user_id, role, mode, content, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                user_id,
                role,
                mode,
                content,
                utc_now()
            )
        )

        conn.commit()

    finally:
        safe_close(conn, cur)


def load_messages(user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT role, content, mode, created_at
            FROM messages
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (user_id, limit)
        )

        rows = list(cur.fetchall())
        rows.reverse()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "mode": row["mode"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]

    finally:
        safe_close(conn, cur)


# ============================================================
# Kurssimateriaalit
# ============================================================

def save_course_material(
    user_id: str,
    filename: str,
    content: str,
    file_hash: str = None
) -> bool:
    """
    Palauttaa True, jos materiaali tallennettiin.
    Palauttaa False, jos sama materiaali oli jo tallennettu.
    """

    ensure_user(user_id)

    if file_hash is None:
        file_hash = hash_text(filename + content)

    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO course_materials (
                user_id,
                filename,
                file_hash,
                content,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, file_hash) DO NOTHING
            RETURNING id
            """,
            (
                user_id,
                filename,
                file_hash,
                content,
                utc_now()
            )
        )

        inserted_row = cur.fetchone()
        inserted = inserted_row is not None

        conn.commit()

        return inserted

    finally:
        safe_close(conn, cur)


def load_course_materials(user_id: str) -> List[Dict[str, Any]]:
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, filename, file_hash, content, created_at
            FROM course_materials
            WHERE user_id = %s
            ORDER BY id DESC
            """,
            (user_id,)
        )

        rows = cur.fetchall()

        return [
            {
                "id": row["id"],
                "filename": row["filename"],
                "file_hash": row["file_hash"],
                "content": row["content"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]

    finally:
        safe_close(conn, cur)


def load_course_context(user_id: str, max_chars: int = 30000) -> str:
    materials = load_course_materials(user_id)

    blocks = []
    total = 0

    for item in materials:
        block = (
            "\n\n--- TIEDOSTO: "
            + str(item["filename"])
            + " | TALLENNETTU: "
            + str(item["created_at"])
            + " ---\n"
            + str(item["content"])
        )

        if total + len(block) > max_chars:
            remaining = max_chars - total

            if remaining > 0:
                blocks.append(block[:remaining])

            break

        blocks.append(block)
        total += len(block)

    return "\n".join(blocks)


def search_course_materials_keyword(
    user_id: str,
    query: str,
    max_results: int = 5,
    chunk_size: int = 1200
) -> str:
    """
    Yksinkertainen avainsanahaku tallennetuista kurssimateriaaleista.
    Tämä on demoversioon sopiva ratkaisu, mutta ei vielä semanttinen haku.
    """

    materials = load_course_materials(user_id)

    query_terms = [
        term.lower().strip(".,;:!?()[]{}\"'")
        for term in query.split()
        if len(term.strip()) > 3
    ]

    general_terms = [
        "tutkimuskysymys",
        "aineisto",
        "menetelmä",
        "analyysi",
        "ohjaaja",
        "arviointi",
        "kriteeri",
        "eettinen",
        "luotettavuus",
        "opinnäytetyö",
        "tutkimussuunnitelma",
        "kirjoittaminen",
        "rajaus"
    ]

    for term in general_terms:
        if term not in query_terms:
            query_terms.append(term)

    scored_chunks = []

    for item in materials:
        content = item["content"] or ""
        filename = item["filename"]

        chunks = [
            content[i:i + chunk_size]
            for i in range(0, len(content), chunk_size)
        ]

        for chunk in chunks:
            lower_chunk = chunk.lower()
            score = 0

            for term in query_terms:
                score += lower_chunk.count(term)

            if score > 0:
                scored_chunks.append({
                    "score": score,
                    "filename": filename,
                    "chunk": chunk
                })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    selected = scored_chunks[:max_results]

    if not selected:
        return ""

    blocks = []

    for item in selected:
        block = (
            "\n\n--- LÄHDE: "
            + str(item["filename"])
            + " | OSUMA: "
            + str(item["score"])
            + " ---\n"
            + str(item["chunk"])
        )

        blocks.append(block)

    return "\n".join(blocks)


def delete_course_material(user_id: str, material_id: int):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            DELETE FROM course_materials
            WHERE user_id = %s AND id = %s
            """,
            (user_id, material_id)
        )

        conn.commit()

    finally:
        safe_close(conn, cur)


def clear_course_materials(user_id: str):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            DELETE FROM course_materials
            WHERE user_id = %s
            """,
            (user_id,)
        )

        conn.commit()

    finally:
        safe_close(conn, cur)


# ============================================================
# Viikkokatsaukset
# ============================================================

def save_checkin(
    user_id: str,
    completed: str,
    blocked: str,
    next_action: str,
    support_needed: str,
    coach_response: str
):
    ensure_user(user_id)

    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO checkins (
                user_id,
                completed,
                blocked,
                next_action,
                support_needed,
                coach_response,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                completed,
                blocked,
                next_action,
                support_needed,
                coach_response,
                utc_now()
            )
        )

        conn.commit()

    finally:
        safe_close(conn, cur)


def load_checkins(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                id,
                completed,
                blocked,
                next_action,
                support_needed,
                coach_response,
                created_at
            FROM checkins
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (user_id, limit)
        )

        rows = cur.fetchall()

        return [
            {
                "id": row["id"],
                "completed": row["completed"],
                "blocked": row["blocked"],
                "next_action": row["next_action"],
                "support_needed": row["support_needed"],
                "coach_response": row["coach_response"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]

    finally:
        safe_close(conn, cur)


# ============================================================
# Tietojen poisto
# ============================================================

def delete_user_data(user_id: str):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM checkins WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM course_materials WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM messages WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM profiles WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))

        conn.commit()

    finally:
        safe_close(conn, cur)


# ============================================================
# Käyttäjäkohtainen yhteenveto
# ============================================================

def get_user_summary(user_id: str) -> dict:
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM messages
            WHERE user_id = %s
            """,
            (user_id,)
        )
        message_count = cur.fetchone()["count"]

        cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM course_materials
            WHERE user_id = %s
            """,
            (user_id,)
        )
        material_count = cur.fetchone()["count"]

        cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM checkins
            WHERE user_id = %s
            """,
            (user_id,)
        )
        checkin_count = cur.fetchone()["count"]

        return {
            "message_count": message_count,
            "material_count": material_count,
            "checkin_count": checkin_count
        }

    finally:
        safe_close(conn, cur)
