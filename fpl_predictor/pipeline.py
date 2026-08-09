import os
import pandas as pd
import db
from dotenv import load_dotenv
import data_loader, features, odds, train, predict

load_dotenv()

BASE_PATH = os.environ.get("FPL_DATA_PATH", r"C:\Users\charl\Fantasy-Premier-League\data")
ODDS_BASE_PATH = os.environ.get("ODDS_DATA_PATH", r"C:\Users\charl\Documents\FPLPredictor\data\odds")

# Completed seasons only — safe to pull from vaastav's repo (finished, won't change).
PAST_SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
PAST_ODDS_SEASONS = ['2122', '2223', '2324', '2425', '2526']

# In-progress season — fetched live from the FPL API instead, each run.
CURRENT_SEASON = '2026-27'


def build_training_data():
    historical_df = data_loader.load_gameweek_data(BASE_PATH, PAST_SEASONS)
    current_df = data_loader.fetch_current_season_gameweek_data(CURRENT_SEASON)

    full_df = pd.concat([historical_df, current_df], ignore_index=True)

    full_df = features.clean_positions(full_df)
    full_df = features.add_rolling_points_avg(full_df)
    full_df = features.add_defcon_features(full_df)
    full_df = features.add_xgi_features(full_df)
    full_df = features.add_clean_sheet_features(full_df)

    team_df = features.build_team_strength(full_df)

    # Historical team lookup from vaastav...
    team_lookup = data_loader.load_team_lookup(BASE_PATH, PAST_SEASONS)

    # ...plus current season's team lookup, from the live FPL API — avoids the
    # opponent_team_name_x/_y merge collision from having two separate sources.
    bootstrap = data_loader.fetch_fpl_bootstrap()
    current_team_lookup = pd.DataFrame(bootstrap['teams'])[['id', 'name']].copy()
    current_team_lookup['season'] = CURRENT_SEASON
    team_lookup = pd.concat([team_lookup, current_team_lookup], ignore_index=True)

    full_df = features.merge_opponent_strength(full_df, team_df, team_lookup)
    full_df = features.add_rolling_expected_stats(full_df)

    season_label_map = dict(zip(PAST_ODDS_SEASONS, PAST_SEASONS))
    all_odds = odds.load_historical_odds(ODDS_BASE_PATH, PAST_ODDS_SEASONS, season_label_map)
    full_df = odds.merge_historical_odds_into_full_df(full_df, all_odds)

    model_df = full_df.dropna(subset=features.FEATURE_COLS + ['total_points']).copy()
    return full_df, team_df, model_df


def run_training():
    _, _, model_df = build_training_data()
    train.train_classifiers(model_df)


def run_weekly_prediction():
    full_df, team_df, _ = build_training_data()
    bootstrap = data_loader.fetch_fpl_bootstrap()
    fixtures_raw = data_loader.fetch_fpl_fixtures()

    next_gw_fixtures, next_gw = predict.build_next_gw_fixtures(bootstrap, fixtures_raw)
    next_week_players = predict.build_next_week_players(full_df, team_df, next_gw_fixtures)

    live_team_odds = odds.get_live_team_odds(os.environ.get("ODDS_API_KEY"))
    next_week_players = predict.attach_live_odds(next_week_players, live_team_odds)

    return predict.predict_next_gameweek(next_week_players), next_gw


if __name__ == "__main__":
    db.init_db()
    predictions, gw = run_weekly_prediction()
    db.log_predictions(predictions, gw)

    print(f"\n=== Top 10 predicted 6+ picks, GW{gw} ===")
    print(predictions[['name', 'position', 'team', 'opponent_team_name', 'prob_6plus']]
          .sort_values('prob_6plus', ascending=False).head(10))

    print(f"\n=== Top 10 predicted 5+ picks, GW{gw} ===")
    print(predictions[['name', 'position', 'team', 'opponent_team_name', 'prob_5plus']]
          .sort_values('prob_5plus', ascending=False).head(10))