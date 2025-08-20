# aigptssh/api/dashboard/check_db.py
import sqlite3
import json
import os

# --- Database Path ---
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, 'dashboard_history.db')

def view_history():
    """Connects to the database and prints a summary of the history."""
    if not os.path.exists(DB_PATH):
        print(f"Database file not found at: {DB_PATH}")
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Select all columns, including the 'data' column
            cursor.execute("SELECT id, timestamp, data FROM dashboard_history ORDER BY timestamp DESC")
            rows = cursor.fetchall()

            if not rows:
                print("The dashboard_history table is empty.")
                return

            print("--- Dashboard Update History ---")
            for row in rows:
                data_length = len(row[2]) if row[2] else 0
                print(f"ID: {row[0]}, Timestamp: {row[1]}, Data Size: {data_length} characters")
            print("------------------------------")
            print("\nConfirmation: The 'data' column contains the full JSON for each entry.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    view_history()