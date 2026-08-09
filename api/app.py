from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os

app = FastAPI(title="FPL Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fplpredictor.vercel.app"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fpl.db')

def query_db(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

@app.get("/predictions/latest")
def latest_predictions(threshold: str = "6plus", limit: int = 10):
    if threshold not in ("5plus", "6plus"):
        raise HTTPException(400, "threshold must be '5plus' or '6plus'")

    latest_gw = query_db("SELECT MAX(gw) as gw FROM predictions_log")[0]['gw']
    if latest_gw is None:
        raise HTTPException(404, "No predictions logged yet")

    rows = query_db(f"""
        SELECT name, team, opponent_team_name, was_home, prob_{threshold}, gw
        FROM predictions_log
        WHERE gw = ?
        ORDER BY prob_{threshold} DESC
        LIMIT ?
    """, (latest_gw, limit))
    return {"gameweek": latest_gw, "threshold": threshold, "predictions": rows}

@app.get("/predictions/gw/{gw}")
def predictions_for_gw(gw: int, threshold: str = "6plus", limit: int = 10):
    if threshold not in ("5plus", "6plus"):
        raise HTTPException(400, "threshold must be '5plus' or '6plus'")
    rows = query_db(f"""
        SELECT name, team, opponent_team_name, was_home, prob_{threshold}, gw
        FROM predictions_log
        WHERE gw = ?
        ORDER BY prob_{threshold} DESC
        LIMIT ?
    """, (gw, limit))
    return {"gameweek": gw, "threshold": threshold, "predictions": rows}

@app.get("/")
def root():
    return {"status": "ok"}