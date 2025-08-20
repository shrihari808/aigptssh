# aigptssh/api/dashboard/history.py
import sqlite3
import json
from datetime import datetime, timezone
import os

# --- Database Path ---
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, 'dashboard_history.db')

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_dashboard_history_table():
    """Creates the dashboard_history table if it doesn't exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL
            );
        """)
        print("INFO: 'dashboard_history' table checked/created successfully.")

def save_dashboard_history(data):
    """Saves the dashboard data to the history table."""
    if not data:
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    data_json = json.dumps(data)

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO dashboard_history (timestamp, data) VALUES (?, ?)",
            (timestamp, data_json)
        )
        print(f"INFO: Saved dashboard data from {timestamp} to history.")

# --- Initialize the database table on module load ---
create_dashboard_history_table()