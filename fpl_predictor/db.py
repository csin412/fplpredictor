import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fpl.db')

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gw INTEGER NOT NULL,
            team TEXT,
            opponent_team_name TEXT,
            was_home INTEGER,
            prob_5plus REAL,
            prob_6plus REAL,
            actual_points INTEGER,
            run_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_predictions(predictions_df, gw):
    conn = get_connection()
    conn.execute("DELETE FROM predictions_log WHERE gw = ?", (gw,))
    conn.commit()

    log_rows = predictions_df[['name', 'team', 'opponent_team_name', 'was_home',
                                 'prob_5plus', 'prob_6plus']].copy()
    log_rows['gw'] = gw
    log_rows.to_sql('predictions_log', conn, if_exists='append', index=False)
    conn.close()
    print(f"Logged {len(log_rows)} predictions for GW{gw} to {DB_PATH}")