"""Authentication and history storage for QuDrugGuard V2."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bcrypt


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                drug_a TEXT NOT NULL,
                drug_b TEXT NOT NULL,
                prediction_label TEXT NOT NULL,
                risk_score REAL NOT NULL,
                confidence REAL NOT NULL,
                shared_enzymes TEXT NOT NULL,
                details_json TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.commit()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def signup(username: str, password: str, full_name: str) -> dict[str, Any]:
    init_db()
    username = username.strip().lower()
    full_name = full_name.strip()
    if len(username) < 3:
        return {"ok": False, "message": "Username must be at least 3 characters."}
    if len(password) < 8:
        return {"ok": False, "message": "Password must be at least 8 characters."}
    if not full_name:
        return {"ok": False, "message": "Full name is required."}
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        with get_connection() as connection:
            cursor = connection.execute("INSERT INTO users (username, full_name, password_hash, created_at) VALUES (?, ?, ?, ?)", (username, full_name, password_hash, _utc_now()))
            connection.commit()
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return {"ok": False, "message": "Username already exists."}
    return {"ok": True, "message": "Signup successful.", "user": {"id": user_id, "username": username, "full_name": full_name}}


def login(username: str, password: str) -> dict[str, Any]:
    init_db()
    username = username.strip().lower()
    with get_connection() as connection:
        row = connection.execute("SELECT id, username, full_name, password_hash, created_at FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return {"ok": False, "message": "User not found."}
    if not bcrypt.checkpw(password.encode("utf-8"), row["password_hash"]):
        return {"ok": False, "message": "Incorrect password."}
    return {"ok": True, "message": "Login successful.", "user": {"id": row["id"], "username": row["username"], "full_name": row["full_name"], "created_at": row["created_at"]}}


def save_check(user_id: int, drug_a: str, drug_b: str, prediction_label: str, risk_score: float, confidence: float, shared_enzymes: list[str], details: dict[str, Any]) -> dict[str, Any]:
    init_db()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO history (
                user_id, drug_a, drug_b, prediction_label, risk_score, confidence,
                shared_enzymes, details_json, checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, drug_a, drug_b, prediction_label, round(float(risk_score), 4), round(float(confidence), 4), ", ".join(shared_enzymes), json.dumps(details, sort_keys=True), _utc_now()),
        )
        connection.commit()
        record_id = cursor.lastrowid
    return {"ok": True, "history_id": record_id}


def get_history(user_id: int, limit: int | None = None) -> list[dict[str, Any]]:
    init_db()
    query = """
        SELECT id, drug_a, drug_b, prediction_label, risk_score, confidence,
               shared_enzymes, details_json, checked_at
        FROM history
        WHERE user_id = ?
        ORDER BY checked_at DESC
    """
    params: tuple[Any, ...] = (user_id,)
    if limit:
        query += " LIMIT ?"
        params = (user_id, limit)
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    history = []
    for row in rows:
        history.append({"id": row["id"], "drug_a": row["drug_a"], "drug_b": row["drug_b"], "prediction_label": row["prediction_label"], "risk_score": row["risk_score"], "confidence": row["confidence"], "shared_enzymes": row["shared_enzymes"], "details": json.loads(row["details_json"]), "checked_at": row["checked_at"]})
    return history


def self_test(verbose: bool = True) -> dict[str, Any]:
    init_db()
    suffix = uuid.uuid4().hex[:8]
    username = f"qudrug_{suffix}"
    password = "Quantum@2026"
    signup_result = signup(username=username, password=password, full_name="QuDrug Test User")
    login_result = login(username=username, password=password)
    wrong_password = login(username=username, password="bad-password")
    passed = signup_result["ok"] and login_result["ok"] and not wrong_password["ok"]
    if verbose:
        print("Signup OK" if signup_result["ok"] else signup_result["message"])
        print("Login OK" if login_result["ok"] else login_result["message"])
        print("Wrong password rejected" if not wrong_password["ok"] else "Wrong password was accepted")
    return {"passed": passed, "signup": signup_result, "login": login_result, "wrong_password": wrong_password}


if __name__ == "__main__":
    self_test(verbose=True)
