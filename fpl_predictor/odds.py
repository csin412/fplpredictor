import pandas as pd
import numpy as np
import requests
import os
from scipy.stats import poisson
from scipy.optimize import brentq

TEAM_NAME_MAP = {
    'Man United': 'Man Utd',
    'Tottenham': 'Spurs',
    'Man Utd': 'Man United',
    'Spurs': 'Tottenham',
    'Sheffield United': 'Sheffield Utd'
}

LIVE_ODDS_TEAM_MAP = {
    'Manchester United': 'Man Utd',
    'Manchester City': 'Man City',
    'Tottenham Hotspur': 'Spurs',
    'Nottingham Forest': "Nott'm Forest",
    'Brighton and Hove Albion': 'Brighton',
    'Leeds United': 'Leeds',
    'Newcastle United': 'Newcastle'
}

def implied_total_goals(prob_over_2_5):
    if pd.isna(prob_over_2_5):
        return np.nan
    def f(lam):
        return poisson.sf(2, lam) - prob_over_2_5
    try:
        return brentq(f, 0.1, 8.0)
    except ValueError:
        return np.nan

def load_historical_odds(base_path, odds_seasons, season_label_map):
    odds_dfs = []
    for s in odds_seasons:
        path = os.path.join(base_path, f'odds{s}.csv')
        d = pd.read_csv(path)
        d['season'] = season_label_map[s]
        odds_dfs.append(d)
    all_odds = pd.concat(odds_dfs, ignore_index=True).copy()
    all_odds['Date'] = pd.to_datetime(all_odds['Date'], dayfirst=True)
    return all_odds

def devig_and_derive_goals(odds_df):
    """Adds win/draw/loss probs + market expected goals. Works on both historical and live odds."""
    implied_H = 1 / odds_df['AvgH']
    implied_D = 1 / odds_df['AvgD']
    implied_A = 1 / odds_df['AvgA']
    overround = implied_H + implied_D + implied_A

    odds_df['prob_home_win'] = implied_H / overround
    odds_df['prob_draw'] = implied_D / overround
    odds_df['prob_away_win'] = implied_A / overround

    implied_over = 1 / odds_df['Avg>2.5']
    implied_under = 1 / odds_df['Avg<2.5']
    ou_overround = implied_over + implied_under
    odds_df['prob_over_2.5'] = implied_over / ou_overround
    odds_df['match_expected_goals'] = odds_df['prob_over_2.5'].apply(implied_total_goals)

    odds_df['home_goal_share'] = odds_df['prob_home_win'] + 0.5 * odds_df['prob_draw']
    odds_df['away_goal_share'] = odds_df['prob_away_win'] + 0.5 * odds_df['prob_draw']
    odds_df['home_expected_goals'] = odds_df['match_expected_goals'] * odds_df['home_goal_share']
    odds_df['away_expected_goals'] = odds_df['match_expected_goals'] * odds_df['away_goal_share']
    return odds_df

def reshape_odds_to_team_level(odds_df, extra_group_cols=None):
    """extra_group_cols e.g. ['season', 'Date'] for historical; None for live (single gameweek)."""
    base_cols = ['HomeTeam_mapped', 'AwayTeam_mapped', 'prob_home_win', 'prob_draw',
                 'prob_away_win', 'home_expected_goals', 'away_expected_goals']
    if extra_group_cols:
        base_cols = extra_group_cols + base_cols

    home_rows = odds_df[base_cols].copy().rename(columns={
        'HomeTeam_mapped': 'team', 'AwayTeam_mapped': 'opponent_team_name',
        'prob_home_win': 'team_win_prob', 'prob_away_win': 'opponent_win_prob',
        'home_expected_goals': 'market_expected_goals_for',
        'away_expected_goals': 'market_expected_goals_against'})
    home_rows['was_home'] = True

    away_rows = odds_df[base_cols].copy().rename(columns={
        'AwayTeam_mapped': 'team', 'HomeTeam_mapped': 'opponent_team_name',
        'prob_away_win': 'team_win_prob', 'prob_home_win': 'opponent_win_prob',
        'away_expected_goals': 'market_expected_goals_for',
        'home_expected_goals': 'market_expected_goals_against'})
    away_rows['was_home'] = False

    team_odds = pd.concat([home_rows, away_rows], ignore_index=True)
    team_odds = team_odds.rename(columns={'prob_draw': 'team_draw_prob'})
    team_odds['market_cs_prob'] = np.exp(-team_odds['market_expected_goals_against'])
    return team_odds

def merge_historical_odds_into_full_df(full_df, all_odds):
    all_odds['HomeTeam_mapped'] = all_odds['HomeTeam'].replace(TEAM_NAME_MAP)
    all_odds['AwayTeam_mapped'] = all_odds['AwayTeam'].replace(TEAM_NAME_MAP)
    all_odds = devig_and_derive_goals(all_odds)

    team_odds = reshape_odds_to_team_level(all_odds, extra_group_cols=['season', 'Date'])
    team_odds['Date_only'] = team_odds['Date'].dt.date

    full_df['kickoff_date'] = pd.to_datetime(full_df['kickoff_time']).dt.date

    full_df = full_df.merge(
        team_odds[['season', 'Date_only', 'team', 'opponent_team_name', 'was_home',
                   'team_win_prob', 'team_draw_prob', 'opponent_win_prob',
                   'market_expected_goals_for', 'market_expected_goals_against', 'market_cs_prob']],
        left_on=['season', 'kickoff_date', 'team', 'opponent_team_name', 'was_home'],
        right_on=['season', 'Date_only', 'team', 'opponent_team_name', 'was_home'],
        how='left'
    )
    print(full_df['market_expected_goals_for'].isna().sum(), "rows failed to match odds")
    return full_df

def fetch_current_odds(api_key):
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    params = {"apiKey": api_key, "regions": "uk", "markets": "h2h,totals",
              "oddsFormat": "decimal", "dateFormat": "iso"}
    response = requests.get(url, params=params)
    print(f"Requests remaining this month: {response.headers.get('x-requests-remaining')}")
    return response.json()

def parse_live_odds(raw_odds):
    rows = []
    for match in raw_odds:
        home_team, away_team = match['home_team'], match['away_team']
        h_odds, d_odds, a_odds, over_odds, under_odds = [], [], [], [], []
        for bookmaker in match['bookmakers']:
            for market in bookmaker['markets']:
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            h_odds.append(outcome['price'])
                        elif outcome['name'] == away_team:
                            a_odds.append(outcome['price'])
                        else:
                            d_odds.append(outcome['price'])
                elif market['key'] == 'totals':
                    for outcome in market['outcomes']:
                        if outcome['point'] == 2.5:
                            (over_odds if outcome['name'] == 'Over' else under_odds).append(outcome['price'])
        rows.append({
            'HomeTeam': home_team, 'AwayTeam': away_team, 'Date': match['commence_time'],
            'AvgH': sum(h_odds)/len(h_odds) if h_odds else None,
            'AvgD': sum(d_odds)/len(d_odds) if d_odds else None,
            'AvgA': sum(a_odds)/len(a_odds) if a_odds else None,
            'Avg>2.5': sum(over_odds)/len(over_odds) if over_odds else None,
            'Avg<2.5': sum(under_odds)/len(under_odds) if under_odds else None,
        })
    return pd.DataFrame(rows)

def get_live_team_odds(api_key):
    """Full pipeline: fetch -> parse -> map names -> devig -> reshape. Returns team-level odds for next GW."""
    raw = fetch_current_odds(api_key)
    live_odds_df = parse_live_odds(raw)
    live_odds_df['HomeTeam_mapped'] = live_odds_df['HomeTeam'].replace(LIVE_ODDS_TEAM_MAP)
    live_odds_df['AwayTeam_mapped'] = live_odds_df['AwayTeam'].replace(LIVE_ODDS_TEAM_MAP)
    live_odds_df = devig_and_derive_goals(live_odds_df)
    return reshape_odds_to_team_level(live_odds_df)