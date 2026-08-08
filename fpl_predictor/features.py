import pandas as pd
import numpy as np

def clean_positions(full_df):
    full_df = full_df[full_df['position'] != 'AM'].copy()
    full_df['position'] = full_df['position'].replace({'GKP': 'GK'})
    return full_df

def add_rolling_points_avg(full_df):
    full_df = full_df.sort_values(['name', 'season', 'round']).reset_index(drop=True)
    full_df['rolling_5_avg'] = full_df.groupby('name')['total_points'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    return full_df

def add_defcon_features(full_df):
    def hit_defcon_threshold(row):
        if pd.isna(row['defensive_contribution']):
            return None
        if row['position'] == 'DEF':
            return 1 if row['defensive_contribution'] >= 10 else 0
        if row['position'] == 'MID':
            return 1 if row['defensive_contribution'] >= 12 else 0
        return 0

    full_df['hit_defcon'] = full_df.apply(hit_defcon_threshold, axis=1)
    full_df['rolling_5_defcon_rate'] = full_df.groupby('name')['hit_defcon'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    ).fillna(0)
    return full_df

def add_xgi_features(full_df):
    goal_points_map = {'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
    assist_points_map = {'GK': 3, 'DEF': 3, 'MID': 3, 'FWD': 3}
    full_df['xG_points'] = full_df['expected_goals'] * full_df['position'].map(goal_points_map)
    full_df['xA_points'] = full_df['expected_assists'] * full_df['position'].map(assist_points_map)
    full_df['xGI_points'] = full_df['xG_points'] + full_df['xA_points']
    full_df['rolling_5_xGI_points'] = full_df.groupby('name')['xGI_points'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    return full_df

def add_clean_sheet_features(full_df):
    clean_sheet_points_map = {'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
    full_df['rolling_5_xGA'] = full_df.groupby('name')['expected_goals_conceded'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    full_df['estimated_cs_prob'] = np.exp(-full_df['rolling_5_xGA'])
    full_df['estimated_cs_points'] = full_df['estimated_cs_prob'] * full_df['position'].map(clean_sheet_points_map)
    return full_df

def build_team_strength(full_df):
    team_rows = []
    for _, row in full_df.drop_duplicates(subset=['season', 'round', 'team', 'was_home']).iterrows():
        if row['was_home']:
            goals_for, goals_against = row['team_h_score'], row['team_a_score']
        else:
            goals_for, goals_against = row['team_a_score'], row['team_h_score']
        team_rows.append({'season': row['season'], 'round': row['round'], 'team': row['team'],
                           'goals_for': goals_for, 'goals_against': goals_against})

    team_df = pd.DataFrame(team_rows).drop_duplicates()
    team_df = team_df.sort_values(['team', 'season', 'round']).reset_index(drop=True)
    team_df['rolling_5_goals_for'] = team_df.groupby('team')['goals_for'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    team_df['rolling_5_goals_against'] = team_df.groupby('team')['goals_against'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    return team_df

def merge_opponent_strength(full_df, team_df, team_lookup):
    team_id_lookup = team_lookup[['season', 'id', 'name']].rename(
        columns={'id': 'opponent_team', 'name': 'opponent_team_name'})
    full_df = full_df.merge(team_id_lookup, on=['season', 'opponent_team'], how='left')

    opponent_strength = team_df.rename(columns={
        'team': 'opponent_team_name',
        'rolling_5_goals_for': 'opponent_rolling_5_goals_for',
        'rolling_5_goals_against': 'opponent_rolling_5_goals_against',
    })[['season', 'round', 'opponent_team_name', 'opponent_rolling_5_goals_for', 'opponent_rolling_5_goals_against']]

    return full_df.merge(opponent_strength, on=['season', 'round', 'opponent_team_name'], how='left')

def add_rolling_expected_stats(full_df):
    for col in ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'minutes']:
        full_df[f'rolling_5_{col}'] = full_df.groupby('name')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    full_df['position'] = full_df['position'].astype('category')
    return full_df

FEATURE_COLS = [
    'rolling_5_avg', 'rolling_5_expected_goals', 'rolling_5_expected_assists',
    'rolling_5_expected_goal_involvements', 'rolling_5_minutes', 'rolling_5_xGI_points',
    'estimated_cs_prob', 'estimated_cs_points', 'opponent_rolling_5_goals_for',
    'opponent_rolling_5_goals_against', 'was_home', 'position',
    'team_win_prob', 'team_draw_prob', 'opponent_win_prob',
    'market_expected_goals_for', 'market_expected_goals_against', 'market_cs_prob'
]