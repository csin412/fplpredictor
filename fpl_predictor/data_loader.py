import os
import time
import pandas as pd
import unicodedata
import requests

SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://fantasy.premierleague.com/',
}


def load_gameweek_data(base_path, seasons=SEASONS):
    all_dfs = []
    for season in seasons:
        path = os.path.join(base_path, season, 'gws', 'merged_gw.csv')
        season_df = pd.read_csv(path)
        season_df['season'] = season
        all_dfs.append(season_df)
    return pd.concat(all_dfs, ignore_index=True)


def load_team_lookup(base_path, seasons=SEASONS):
    team_lookup_dfs = []
    for season in seasons:
        path = os.path.join(base_path, season, 'teams.csv')
        t = pd.read_csv(path)
        t['season'] = season
        team_lookup_dfs.append(t)
    return pd.concat(team_lookup_dfs, ignore_index=True)


def _get_with_retry(session, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
        except requests.exceptions.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"Request error on {url}: {e}. Retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 429):
            wait = 5 * (attempt + 1)
            print(f"Got {resp.status_code} from {url}, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        else:
            raise RuntimeError(f"Request to {url} failed: {resp.status_code} - {resp.text[:200]}")

    raise RuntimeError(f"Request to {url} failed after {max_retries} attempts (timeouts/403/429)")


def fetch_fpl_bootstrap():
    with requests.Session() as session:
        resp = _get_with_retry(session, "https://fantasy.premierleague.com/api/bootstrap-static/")
        return resp.json()


def fetch_fpl_fixtures():
    with requests.Session() as session:
        resp = _get_with_retry(session, "https://fantasy.premierleague.com/api/fixtures/")
        return resp.json()


def fetch_current_season_gameweek_data(season_label='2026-27'):
    bootstrap = fetch_fpl_bootstrap()
    players = pd.DataFrame(bootstrap['elements'])
    teams = pd.DataFrame(bootstrap['teams'])[['id', 'name']].rename(columns={'id': 'team_id', 'name': 'team_name'})
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    players = players.merge(teams, left_on='team', right_on='team_id')
    players['position'] = players['element_type'].map(position_map)

    all_rows = []
    failed_ids = []

    expected_cols = ['name', 'position', 'team', 'season', 'round', 'opponent_team', 'was_home',
                      'total_points', 'minutes', 'expected_goals', 'expected_assists',
                      'expected_goal_involvements', 'expected_goals_conceded',
                      'defensive_contribution', 'kickoff_time', 'team_h_score', 'team_a_score']

    with requests.Session() as session:
        for i, (_, player) in enumerate(players.iterrows()):
            url = f"https://fantasy.premierleague.com/api/element-summary/{player['id']}/"
            try:
                resp = _get_with_retry(session, url, max_retries=2)
            except RuntimeError:
                failed_ids.append(player['id'])
                continue

            history = resp.json().get('history', [])
            for gw in history:
                all_rows.append({
                    'name': f"{player['first_name']} {player['second_name']}",
                    'position': player['position'],
                    'team': player['team_name'],
                    'season': season_label,
                    'round': gw['round'],
                    'opponent_team': gw['opponent_team'],   # raw id — resolved to a name later, once, by merge_opponent_strength
                    'was_home': gw['was_home'],
                    'total_points': gw['total_points'],
                    'minutes': gw['minutes'],
                    'expected_goals': gw.get('expected_goals'),
                    'expected_assists': gw.get('expected_assists'),
                    'expected_goal_involvements': gw.get('expected_goal_involvements'),
                    'expected_goals_conceded': gw.get('expected_goals_conceded'),
                    'defensive_contribution': gw.get('defensive_contribution'),
                    'kickoff_time': gw['kickoff_time'],
                    'team_h_score': gw['team_h_score'],
                    'team_a_score': gw['team_a_score'],
                })

            if (i + 1) % 100 == 0:
                print(f"Fetched {i+1}/{len(players)} players...")
                pd.DataFrame(all_rows).to_csv('data/current_season_partial.csv', index=False)
            time.sleep(0.05)

    if failed_ids:
        print(f"Warning: {len(failed_ids)} player requests failed permanently "
              f"(IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''})")

    if not all_rows:
        print(f"No completed gameweeks found for {season_label} yet — season likely hasn't started.")
        return pd.DataFrame(columns=expected_cols)

    return pd.DataFrame(all_rows)

def get_current_player_teams(bootstrap):
    """Maps each player's full name to their CURRENT team, straight from bootstrap.
    Used to override stale historical team assignments after transfers."""
    players = pd.DataFrame(bootstrap['elements'])
    teams = pd.DataFrame(bootstrap['teams'])[['id', 'name']].rename(columns={'id': 'team_id', 'name': 'team_name'})
    players = players.merge(teams, left_on='team', right_on='team_id')
    players['name'] = players['first_name'] + ' ' + players['second_name']
    players['price'] = players['now_cost'] / 10
    return players[['name', 'team_name', 'price']].rename(columns={'team_name': 'team'})

def normalize_name(name):
    if pd.isna(name):
        return name
    n = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
    return ' '.join(n.lower().split())