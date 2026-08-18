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
            position TEXT,
            team TEXT,
            opponent_team_name TEXT,
            was_home INTEGER,
            prob_5plus REAL,
            prob_6plus REAL,
            actual_points INTEGER,
            run_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, gw)
        )
    """)
    conn.commit()
    conn.close()

def migrate_add_unique_constraint():
    conn = get_connection()
    conn.executescript("""
        ALTER TABLE predictions_log RENAME TO predictions_log_old;
        CREATE TABLE predictions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, gw INTEGER NOT NULL, team TEXT,
            opponent_team_name TEXT, was_home INTEGER,
            prob_5plus REAL, prob_6plus REAL, actual_points INTEGER,
            run_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, gw)
        );
        INSERT INTO predictions_log (name, gw, team, opponent_team_name, was_home,
                                      prob_5plus, prob_6plus, actual_points, run_timestamp)
        SELECT name, gw, team, opponent_team_name, was_home,
               prob_5plus, prob_6plus, actual_points, run_timestamp
        FROM predictions_log_old
        GROUP BY name, gw HAVING MAX(run_timestamp);
        DROP TABLE predictions_log_old;
    """)
    conn.commit()
    conn.close()

def migrate_add_position_column():
    conn = get_connection()
    conn.execute("ALTER TABLE predictions_log ADD COLUMN position TEXT")
    conn.commit()
    conn.close()

def log_predictions(predictions_df, gw):
    conn = get_connection()
    log_rows = predictions_df[['name', 'team', 'position', 'opponent_team_name', 'was_home',
                                 'prob_5plus', 'prob_6plus']].drop_duplicates(subset=['name']).copy()
    log_rows['gw'] = gw

    conn.executemany("""
        INSERT INTO predictions_log (name, gw, position, team, opponent_team_name, was_home, prob_5plus, prob_6plus, run_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(name, gw) DO UPDATE SET
            position=excluded.position, team=excluded.team, opponent_team_name=excluded.opponent_team_name,
            was_home=excluded.was_home, prob_5plus=excluded.prob_5plus,
            prob_6plus=excluded.prob_6plus, run_timestamp=excluded.run_timestamp
    """, log_rows[['name', 'gw', 'position', 'team', 'opponent_team_name', 'was_home', 'prob_5plus', 'prob_6plus']].values.tolist())
    conn.commit()
    conn.close()
    print(f"Logged {len(log_rows)} predictions for GW{gw} to {DB_PATH}")

def delete_dupes():
    """Deletes duplicate rows in the predictions_log table, keeping only the most recent run for each player and gameweek."""
    conn = get_connection()
    df = pd.read_sql('SELECT * FROM predictions_log', conn)
    df_deduped = df.sort_values('run_timestamp').drop_duplicates(subset=['name', 'gw'], keep='last')
    conn.execute('DELETE FROM predictions_log')
    df_deduped.to_sql('predictions_log', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    print(f'Before: {len(df)}, After: {len(df_deduped)}')