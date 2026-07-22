import os
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fraud.db"


def get_connection():
    os.makedirs(DB_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            step INTEGER,
            type TEXT,
            amount REAL,
            nameOrig TEXT,
            oldbalanceOrg REAL,
            newbalanceOrig REAL,
            nameDest TEXT,
            oldbalanceDest REAL,
            newbalanceDest REAL,
            fraud_probability REAL,
            is_fraud_predicted INTEGER,
            status TEXT DEFAULT 'PENDING',
            ground_truth INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Reviews table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            action TEXT,
            analyst_note TEXT,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def insert_transaction(tx_dict: dict, proba: float, pred: int, tx_id: str = None) -> str:
    import uuid
    if not tx_id:
        tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions (
            transaction_id, step, type, amount, nameOrig, oldbalanceOrg,
            newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest,
            fraud_probability, is_fraud_predicted, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
    """, (
        tx_id,
        tx_dict.get("step", 1),
        tx_dict.get("type", "TRANSFER"),
        float(tx_dict.get("amount", 0.0)),
        tx_dict.get("nameOrig", "C0000000"),
        float(tx_dict.get("oldbalanceOrg", 0.0)),
        float(tx_dict.get("newbalanceOrig", 0.0)),
        tx_dict.get("nameDest", "M0000000"),
        float(tx_dict.get("oldbalanceDest", 0.0)),
        float(tx_dict.get("newbalanceDest", 0.0)),
        float(proba),
        int(pred)
    ))
    conn.commit()
    conn.close()
    return tx_id


def get_pending_queue(limit: int = 50):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM transactions
        WHERE status = 'PENDING'
        ORDER BY fraud_probability DESC, created_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def review_transaction(tx_id: str, action: str, note: str = ""):
    ground_truth = 1 if action == "DECLINED" else (0 if action == "APPROVED" else None)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE transactions
        SET status = ?, ground_truth = ?
        WHERE transaction_id = ?
    """, (action, ground_truth, tx_id))

    cursor.execute("""
        INSERT INTO reviews (transaction_id, action, analyst_note)
        VALUES (?, ?, ?)
    """, (tx_id, action, note))

    conn.commit()
    conn.close()


def get_reviewed_transactions():
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT * FROM transactions WHERE status IN ('APPROVED', 'DECLINED') AND ground_truth IS NOT NULL
    """, conn)
    conn.close()
    return df


def get_all_transactions_df():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    return df


# Initialize on import
init_db()
