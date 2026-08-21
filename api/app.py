from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import team_selector

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

def _latest_gw():
    latest_gw = query_db("SELECT MAX(gw) as gw FROM predictions_log")[0]['gw']
    if latest_gw is None:
        raise HTTPException(404, "No predictions logged yet")
    return latest_gw

@app.get("/predictions/latest")
def latest_predictions(threshold: str = "6plus", limit: int = 10, position: str | None = None):
    _validate(threshold, position, limit)

    latest_gw = _latest_gw()

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

@app.get("/predictions/search")
def search_player(name: str, threshold:str = "6plus"):
    if threshold not in ("5plus", "6plus"):
        raise HTTPException(400, "threshold must be '5plus' or '6plus'")
    name = name.strip()
    if len(name) < 2:
        raise HTTPException(400, "name must be at least 2 characters long")

    latest_gw = _latest_gw()
    rows = query_db(f"""
        SELECT name, team, position, opponent_team_name, was_home, price, prob_5plus, prob_6plus, gw
        FROM predictions_log
        WHERE gw = ? AND price IS NOT NULL AND name LIKE ?
        ORDER BY prob_{threshold} DESC
        LIMIT 10
    """, (latest_gw, f"%{name}%"))
    return {"gameweek": latest_gw, "query": name, "matches": rows}

@app.get("/team-of-week")
def team_of_week(threshold: str = "6plus"):
    if threshold not in ("5plus", "6plus"):
        raise HTTPException(400, "threshold must be '5plus' or '6plus'")

    latest_gw = _latest_gw()
    rows = query_db(f"""
        SELECT name, team, position, opponent_team_name, was_home, price, prob_5plus, prob_6plus
        FROM predictions_log
        WHERE gw = ? AND price IS NOT NULL
        ORDER BY prob_{threshold} DESC
    """, (latest_gw,))

    result = team_selector.build_team_of_week(rows, threshold)
    if result is None:
        raise HTTPException(404, "No valid team could be built for the latest gameweek")
    return {
        "gameweek": latest_gw,
        "threshold": threshold,
        "formation": result["formation"],
        "total_expected_probability": result["total_prob"],
        "players": result["players"]
    }

@app.get("/")
def root():
    return {"status": "ok"}