from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os

app = FastAPI(title="FPL Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fplpredictor-lovat.vercel.app"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fpl.db')
VALID_POSITIONS = {"GK", "DEF", "MID", "FWD"}
MAX_LIMIT = 25

def _validate(threshold, position, limit):
    if threshold not in ("5plus", "6plus"):
        raise HTTPException(400, "threshold must be '5plus' or '6plus'")
    if position is not None and position not in VALID_POSITIONS:
        raise HTTPException(400, f"position must be one of {VALID_POSITIONS}")
    if limit not in (10, 15, 20, 25):
        raise HTTPException(400, f"limit must be one of 10, 15, 20, 25")

def query_db(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

@app.get("/predictions/latest")
def latest_predictions(threshold: str = "6plus", limit: int = 10, position: str | None = None):
    _validate(threshold, position, limit)

    latest_gw = query_db("SELECT MAX(gw) as gw FROM predictions_log")[0]['gw']
    if latest_gw is None:
        raise HTTPException(404, "No predictions logged yet")

    where = "WHERE gw = ? AND price IS NOT NULL"
    params = [latest_gw]
    if position:
        where += " AND position = ?"
        params.append(position)
    params.append(limit)

    rows = query_db(f"""
        SELECT name, team, position, opponent_team_name, was_home, price, prob_{threshold}, gw
        FROM predictions_log
        {where}
        ORDER BY prob_{threshold} DESC
        LIMIT ?
    """, tuple(params))
    return {"gameweek": latest_gw, "threshold": threshold, "position": position, "predictions": rows}

@app.get("/predictions/gw/{gw}")
def predictions_for_gw(gw: int, threshold: str = "6plus", limit: int = 10, position: str | None = None):
    _validate(threshold, position, limit)
    where = "WHERE gw = ? AND price IS NOT NULL"
    params = [gw]
    if position:
        where += " AND position = ?"
        params.append(position)
    params.append(limit)

    rows = query_db(f"""
        SELECT name, team, position, opponent_team_name, was_home, price, prob_{threshold}, gw
        FROM predictions_log
        {where}
        ORDER BY prob_{threshold} DESC
        LIMIT ?
    """, tuple(params))
    return {"gameweek": gw, "threshold": threshold, "position": position, "predictions": rows}

@app.get("/")
def root():
    return {"status": "ok"}