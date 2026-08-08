import pandas as pd
import requests

SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

def load_gameweek_data(base_path, seasons=SEASONS):
    all_dfs = []
    for season in seasons:
        path = f"{base_path}\\{season}\\gws\\merged_gw.csv"
        season_df = pd.read_csv(path)
        season_df['season'] = season
        all_dfs.append(season_df)
    return pd.concat(all_dfs, ignore_index=True)

def load_team_lookup(base_path, seasons=SEASONS):
    team_lookup_dfs = []
    for season in seasons:
        path = f"{base_path}\\{season}\\teams.csv"
        t = pd.read_csv(path)
        t['season'] = season
        team_lookup_dfs.append(t)
    return pd.concat(team_lookup_dfs, ignore_index=True)

def fetch_fpl_bootstrap():
    return requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()

def fetch_fpl_fixtures():
    return requests.get("https://fantasy.premierleague.com/api/fixtures/").json()