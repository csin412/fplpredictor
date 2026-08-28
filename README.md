# FPL Predictor

A full-stack machine learning app that predicts each Fantasy Premier League (FPL) player's probability of scoring **5+** or **6+** points in the upcoming gameweek — and picks an optimal "Team of the Week" from those predictions.

**Live demo:** [fplpredictor-lovat.vercel.app](https://fplpredictor-lovat.vercel.app)
**API:** [fpl-predictor-api-3njm.onrender.com](https://fpl-predictor-api-3njm.onrender.com)

---

## What it does

- **Predicts point thresholds, not exact scores.** For every player in the next gameweek, two LightGBM classifiers estimate `P(points ≥ 5)` and `P(points ≥ 6)`, which is more robust than trying to predict exact scores.
- **Blends form, fixtures, and market signal.** Features combine each player's rolling underlying stats (xG, xA, defensive contributions), team/opponent strength (rolling goals for/against), and betting odds (implied win/draw probabilities and market-implied expected goals).
- **Refreshes automatically.** A weekly pipeline pulls the latest player data from the official FPL API, merges it with historical season data, attaches live odds, and re-runs predictions for the next fixture list.
- **Builds a Team of the Week.** A constraint-based selector picks the highest-probability valid XI across all legal FPL formations and restraints.
- **Serves it through a simple web UI** with three views: top predictions (filterable by position, sortable, groupable), player search, and the Team of the Week pitch view.

## Architecture

```
FPLPredictor/
├── fpl_predictor/        # Data pipeline, feature engineering, and model training
│   ├── data_loader.py    # Pulls historical (vaastav's FPL archive) + live FPL API data
│   ├── features.py       # Rolling form, xG/xA points, clean sheet probability, opponent strength
│   ├── odds.py           # Devigs betting odds into win/draw/goal probabilities (Poisson-based)
│   ├── pipeline.py       # Orchestrates: build training data -> train models -> weekly predict
│   ├── train.py          # Trains LightGBM classifiers for 5+ and 6+ thresholds
│   ├── predict.py        # Builds next-gameweek player/fixture rows and runs inference
│   └── db.py             # SQLite logging of predictions per gameweek
├── api/                  # FastAPI backend, deployed on Render
│   ├── app.py             # REST endpoints (latest predictions, search, team of the week)
│   └── team_selector.py   # Greedy formation-constrained team-of-the-week selector
└── frontend/              # Static JS/HTML frontend, deployed on Vercel
    └── index.html
```

**Data flow:**

1. `data_loader.py` fetches five completed seasons of gameweek-by-gameweek player data (via [vaastav's FPL historical dataset](https://github.com/vaastav/Fantasy-Premier-League)) plus the current season live from the official FPL API.
2. `features.py` builds rolling-5-gameweek features per player (points, xG/xA, defensive contribution rate, expected clean sheet probability) and per-team strength (rolling goals for/against).
3. `odds.py` pulls historical and live bookmaker odds, removes the bookmaker margin, and derives match/team-level expected goals and win probabilities via a Poisson goals model.
4. `train.py` trains two `LGBMClassifier` models — one for `points ≥ 5`, one for `points ≥ 6` — on all completed seasons.
5. Each week, `pipeline.py` fetches the next fixture list, builds a feature row per player for their upcoming match, attaches live odds, and runs both classifiers to get probabilities.
6. Predictions are logged to a SQLite database (`db.py`) and served via the FastAPI backend for the frontend to consume.

## Tech stack

| Layer | Tools |
|---|---|
| Modeling | LightGBM, scikit-learn-style classifiers |
| Data processing | pandas, NumPy, SciPy (Poisson/root-finding for odds devigging) |
| Data sources | Official FPL API, [vaastav's FPL historical archive](https://github.com/vaastav/Fantasy-Premier-League), The Odds API |
| Backend | FastAPI, SQLite |
| Frontend | JavaScript, HTML/CSS |
| Deployment | Render (API), Vercel (frontend) |

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /predictions/latest?threshold=&position=&limit=` | Top predicted players for the latest logged gameweek |
| `GET /predictions/gw/{gw}?threshold=&position=&limit=` | Predictions for a specific gameweek |
| `GET /predictions/search?name=&threshold=` | Search predictions by player name |
| `GET /team-of-week?threshold=` | Optimal Team of the Week for the latest gameweek |

`threshold` is `5plus` or `6plus`. `position` is one of `GK`, `DEF`, `MID`, `FWD`.

## Team of the Week algorithm

FPL squads must field 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD (10 outfield players), with no more than 3 players from the same club. `team_selector.py` greedily fills each of the legal formations (highest-probability players first), then returns the formation with the highest total expected probability. This isn't a guaranteed global optimum but is a fast, close approximation.

## Running locally

**Prerequisites:** Python 3.10+, a local copy of the [vaastav FPL historical dataset](https://github.com/vaastav/Fantasy-Premier-League), and (optionally) an [Odds API](https://the-odds-api.com/) key for live odds.

```bash
git clone https://github.com/<your-username>/FPLPredictor.git
cd FPLPredictor
pip install -r requirements.txt
```

Create a `.env` file in `fpl_predictor/`:

```
FPL_DATA_PATH=/path/to/Fantasy-Premier-League/data
ODDS_DATA_PATH=/path/to/historical/odds
ODDS_API_KEY=your_odds_api_key
```

Train the models:

```bash
cd fpl_predictor
python -c "import pipeline; pipeline.run_training()"
```

Run a weekly prediction (also logs to the local SQLite DB):

```bash
python pipeline.py
```

Run the API locally:

```bash
cd api
uvicorn app:app --reload
```

Then open `frontend/index.html` in a browser (update `API_URL` in the script if running against a local API instance).

---

Built by [Nok Hei (Charlie) Sin](mailto:charliesin412@gmail.com)
