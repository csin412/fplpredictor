"""
Builds a valid FPL "Team of the Week": the 11 players with the highest predicted
probability of hitting the points threshold, subject to real FPL squad rules:
  - exactly 1 GK
  - 3-5 DEF, 2-5 MID, 1-3 FWD (10 outfield players total)
  - no more than 3 players from the same real-life team

Selection is a greedy fill per formation (highest probability first, skipping
anyone who'd breach the 3-per-team cap), then the formation with the highest
total probability wins. This isn't a guaranteed global optimum (a true optimum
would need an ILP solver) but it's a very close, fast approximation and is
exactly how most "team of the week" tools work in practice.
"""

POSITIONS = ('GK', 'DEF', 'MID', 'FWD')

# All valid outfield splits: DEF 3-5, MID 2-5, FWD 1-3, summing to 10.
FORMATIONS = [
    (d, m, f)
    for d in range(3, 6)
    for m in range(2, 6)
    for f in range(1, 4)
    if d + m + f == 10
]


def _pick(candidates, count, team_counts, used_names):
    selected = []
    for p in candidates:
        if len(selected) >= count:
            break
        if p['name'] in used_names:
            continue
        if team_counts.get(p['team'], 0) >= 3:
            continue
        selected.append(p)
        used_names.add(p['name'])
        team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
    return selected


def _select_for_formation(players_by_pos, formation, prob_key):
    d, m, f = formation
    team_counts = {}
    used_names = set()

    gk = _pick(players_by_pos['GK'], 1, team_counts, used_names)
    defs = _pick(players_by_pos['DEF'], d, team_counts, used_names)
    mids = _pick(players_by_pos['MID'], m, team_counts, used_names)
    fwds = _pick(players_by_pos['FWD'], f, team_counts, used_names)

    squad = gk + defs + mids + fwds
    if len(squad) != 11:
        return None  # not enough eligible players to fill this formation

    return {
        'formation': f"{d}-{m}-{f}",
        'players': squad,
        'total_prob': sum(p[prob_key] for p in squad),
    }


def build_team_of_week(rows, threshold):
    """rows: list of dicts with at least name, team, position, and prob_{threshold}."""
    prob_key = f'prob_{threshold}'
    players_by_pos = {
        pos: sorted((r for r in rows if r['position'] == pos), key=lambda r: r[prob_key], reverse=True)
        for pos in POSITIONS
    }

    best = None
    for formation in FORMATIONS:
        result = _select_for_formation(players_by_pos, formation, prob_key)
        if result and (best is None or result['total_prob'] > best['total_prob']):
            best = result
    return best