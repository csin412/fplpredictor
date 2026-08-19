import joblib
import pandas as pd
import data_loader
from features import FEATURE_COLS

def build_next_gw_fixtures(bootstrap, fixtures_raw):
    teams_lookup = pd.DataFrame(bootstrap['teams'])[['id', 'name']].rename(
        columns={'id': 'team_id', 'name': 'team_name'})
    events_df = pd.DataFrame(bootstrap['events'])
    next_gw = events_df[events_df['is_next'] == True]['id'].values[0]

    fixtures_df = pd.DataFrame(fixtures_raw)
    next_fixtures = fixtures_df[fixtures_df['event'] == next_gw][['team_h', 'team_a', 'kickoff_time']]
    next_fixtures = next_fixtures.merge(
        teams_lookup.rename(columns={'team_id': 'team_h', 'team_name': 'home_team'}), on='team_h')
    next_fixtures = next_fixtures.merge(
        teams_lookup.rename(columns={'team_id': 'team_a', 'team_name': 'away_team'}), on='team_a')

    home_side = next_fixtures[['home_team', 'away_team']].rename(
        columns={'home_team': 'team', 'away_team': 'opponent_team_name'})
    home_side['was_home'] = True
    away_side = next_fixtures[['home_team', 'away_team']].rename(
        columns={'away_team': 'team', 'home_team': 'opponent_team_name'})
    away_side['was_home'] = False

    return pd.concat([home_side, away_side], ignore_index=True), next_gw

def build_next_week_players(full_df, team_df, next_gw_fixtures, current_team_lookup=None):
    latest_season = full_df['season'].max()
    latest_player_rows = (
        full_df[full_df['season'] == latest_season]
        .sort_values(['name', 'round']).groupby('name').tail(1).copy()
    )

    if current_team_lookup is not None:
        lookup = current_team_lookup[['name', 'team', 'price']].rename(columns={'team': 'current_team'}).copy()
        lookup['_key'] = lookup['name'].map(data_loader.normalize_name)
        lookup = lookup.drop_duplicates(subset='_key')  # guard against name collisions

        latest_player_rows['_key'] = latest_player_rows['name'].map(data_loader.normalize_name)
        latest_player_rows = latest_player_rows.merge(lookup[['_key', 'current_team', 'price']], on='_key', how='left')

        unmatched = latest_player_rows['current_team'].isna()
        if unmatched.any():
            print(f"Warning: {unmatched.sum()} players had no current-team match, "
                f"keeping last-known team: {latest_player_rows.loc[unmatched, 'name'].tolist()[:15]}")

        latest_player_rows['team'] = latest_player_rows['current_team'].fillna(latest_player_rows['team'])
        latest_player_rows = latest_player_rows.drop(columns=['current_team', '_key'])

    next_week_players = latest_player_rows.drop(columns=['opponent_team_name', 'was_home']).merge(
        next_gw_fixtures, on='team', how='inner')

    latest_team_stats = (
        team_df.sort_values(['team', 'season', 'round']).groupby('team').tail(1)
        [['team', 'rolling_5_goals_for', 'rolling_5_goals_against']]
        .rename(columns={'team': 'opponent_team_name',
                          'rolling_5_goals_for': 'opponent_rolling_5_goals_for',
                          'rolling_5_goals_against': 'opponent_rolling_5_goals_against'})
    )
    next_week_players = next_week_players.drop(
        columns=['opponent_rolling_5_goals_for', 'opponent_rolling_5_goals_against']
    ).merge(latest_team_stats, on='opponent_team_name', how='left')
    return next_week_players

def attach_live_odds(next_week_players, live_team_odds):
    stale_cols = ['team_win_prob', 'team_draw_prob', 'opponent_win_prob',
                  'market_expected_goals_for', 'market_expected_goals_against', 'market_cs_prob']
    next_week_players = next_week_players.drop(columns=[c for c in stale_cols if c in next_week_players.columns])
    next_week_players = next_week_players.merge(
        live_team_odds[['team', 'opponent_team_name', 'was_home'] + stale_cols],
        on=['team', 'opponent_team_name', 'was_home'], how='left'
    )
    print(next_week_players['team_win_prob'].isna().sum(), "players missing odds data")
    return next_week_players

def predict_next_gameweek(next_week_players, model_dir='models'):
    clf_5 = joblib.load(f'{model_dir}/clf_5plus.pkl')
    clf_6 = joblib.load(f'{model_dir}/clf_6plus.pkl')

    predictable = next_week_players.dropna(subset=FEATURE_COLS).copy()
    print(f"{len(predictable)} of {len(next_week_players)} players have complete features")
    predictable['prob_5plus'] = clf_5.predict_proba(predictable[FEATURE_COLS])[:, 1]
    predictable['prob_6plus'] = clf_6.predict_proba(predictable[FEATURE_COLS])[:, 1]
    return predictable