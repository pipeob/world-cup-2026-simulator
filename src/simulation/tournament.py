import numpy as np
import joblib

COLUMNS = joblib.load("trained_models/columns.pkl")


# GENERAR MARCADOR REALISTA

def generate_score(result):
    """
    Genera un marcador realista con más variedad.
    Basado en distribuciones de goles reales en mundiales.
    """
    if result == 2:  # gana equipo A
        goals_a = np.random.choice([1, 2, 3, 4, 5], p=[0.38, 0.30, 0.18, 0.09, 0.05])
        goals_b = np.random.choice([0, 1, 2, 3],    p=[0.45, 0.32, 0.16, 0.07])
        if goals_b >= goals_a:
            goals_b = goals_a - 1

    elif result == 0:  # gana equipo B
        goals_b = np.random.choice([1, 2, 3, 4, 5], p=[0.38, 0.30, 0.18, 0.09, 0.05])
        goals_a = np.random.choice([0, 1, 2, 3],    p=[0.45, 0.32, 0.16, 0.07])
        if goals_a >= goals_b:
            goals_a = goals_b - 1

    else:  # empate
        g = np.random.choice([0, 1, 2, 3], p=[0.25, 0.45, 0.22, 0.08])
        goals_a = g
        goals_b = g

    return int(goals_a), int(goals_b)


# FEATURES (ALINEADAS AL MODELO)

def build_match_features(team_a, team_b, features_dict):
    a = features_dict[team_a]
    b = features_dict[team_b]

    feature_map = {
        "home_elo":           a["elo"],
        "away_elo":           b["elo"],
        "home_form_points":   a["form_points"],
        "away_form_points":   b["form_points"],
        "home_goals_for":     a["goals_for"],
        "home_goals_against": a["goals_against"],
        "away_goals_for":     b["goals_for"],
        "away_goals_against": b["goals_against"],
        "elo_diff":           a["elo"] - b["elo"],
        "form_points_diff":   a["form_points"] - b["form_points"],
        "goal_diff_recent": (
            (a["goals_for"] - a["goals_against"]) -
            (b["goals_for"] - b["goals_against"])
        )
    }

    return [feature_map[col] for col in COLUMNS]


# SIMULAR PARTIDO

def simulate_match(model, team_a, team_b, features_dict):
    X = build_match_features(team_a, team_b, features_dict)
    probs = model.predict_proba([X])[0]

    # probs: [away_win=0, draw=1, home_win=2]
    result = np.random.choice([0, 1, 2], p=probs)
    goals_a, goals_b = generate_score(result)

    return result, probs, goals_a, goals_b


# FASE DE GRUPOS

def simulate_group(group_teams, features_dict, model):
    standings = {team: {"pts": 0, "gf": 0, "ga": 0} for team in group_teams}
    matches = []

    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            team_a = group_teams[i]
            team_b = group_teams[j]

            res, _, ga, gb = simulate_match(model, team_a, team_b, features_dict)

            matches.append(f"{team_a} {ga} - {gb} {team_b}")

            standings[team_a]["gf"] += ga
            standings[team_a]["ga"] += gb
            standings[team_b]["gf"] += gb
            standings[team_b]["ga"] += ga

            if res == 2:      # gana A
                standings[team_a]["pts"] += 3
            elif res == 0:    # gana B
                standings[team_b]["pts"] += 3
            else:             # empate
                standings[team_a]["pts"] += 1
                standings[team_b]["pts"] += 1

    # Ordenar: puntos → diferencia de gol → goles a favor
    sorted_standings = sorted(
        standings.items(),
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
        reverse=True
    )

    # Devolver lista de (equipo, puntos, gf, ga)
    return sorted_standings, matches


# CLASIFICADOS (32 equipos de 12 grupos)

def get_qualified(groups_results):
    """
    Mundial 2026: 12 grupos de 4 equipos.
    - Top 2 de cada grupo clasifican (24 equipos)
    - Los 8 mejores terceros clasifican (8 equipos)
    Total: 32 equipos al Rd de 32
    """
    first_places  = []
    second_places = []
    third_places  = []

    for group in groups_results:
        # group = [(team, {pts, gf, ga}), ...]
        first_places.append(group[0])
        second_places.append(group[1])
        if len(group) > 2:
            third_places.append(group[2])

    # Mejores terceros por puntos --> dif. goles --> goles a favor
    third_sorted = sorted(
        third_places,
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
        reverse=True
    )
    best_thirds = third_sorted[:8]

    qualified_full = first_places + second_places + best_thirds

    return [team for team, _ in qualified_full]


# RONDA KO

def simulate_knockout(teams, features_dict, model):
    """
    Simula la fase eliminatoria completa.
    Devuelve el campeón y un dict con los resultados de cada ronda.

    Estructura de retorno:
    ko_results = {
        "Ronda de 32": [
            {"team_a": str, "team_b": str, "score_a": int, "score_b": int,
             "winner": str, "penalties": bool},
            ...
        ],
        "Octavos de Final": [...],
        ...
    }
    """
    round_names = [
        "Ronda de 32",      # 32 --> 16
        "Octavos de Final", # 16 --> 8
        "Cuartos de Final", # 8  --> 4
        "Semifinales",      # 4  --> 2
        "Final",            # 2  --> 1
    ]

    ko_results = {}
    round_num = 0

    while len(teams) > 1:
        round_name = round_names[round_num] if round_num < len(round_names) else f"Ronda {round_num + 1}"
        round_matches = []
        winners = []

        for i in range(0, len(teams), 2):
            team_a = teams[i]
            team_b = teams[i + 1]

            res, _, ga, gb = simulate_match(model, team_a, team_b, features_dict)
            penalties = False

            if res == 2:
                winner = team_a
            elif res == 0:
                winner = team_b
            else:
                # Empate en 90 min ---> penales con ventaja proporcional al ELO
                elo_a = features_dict[team_a]["elo"]
                elo_b = features_dict[team_b]["elo"]
                p_a = elo_a / (elo_a + elo_b)
                winner = np.random.choice([team_a, team_b], p=[p_a, 1 - p_a])
                penalties = True

            round_matches.append({
                "team_a":   team_a,
                "team_b":   team_b,
                "score_a":  ga,
                "score_b":  gb,
                "winner":   winner,
                "penalties": penalties,
            })
            winners.append(winner)

        ko_results[round_name] = round_matches
        teams = winners
        round_num += 1

    champion = teams[0]
    return champion, ko_results


# ORNEO COMPLETO

def simulate_tournament(groups, features_dict, model):
    groups_results = []
    all_matches = {}
    qualified_by_group = {}

    for group_name, teams in groups.items():
        standings, matches = simulate_group(teams, features_dict, model)
        groups_results.append(standings)
        all_matches[group_name] = matches
        qualified_by_group[group_name] = [
            (team, data["pts"], data["gf"], data["ga"])
            for team, data in standings
        ]

    qualified = get_qualified(groups_results)

    if len(qualified) != 32:
        raise ValueError(f"Se esperaban 32 equipos, se obtuvieron {len(qualified)}")

    champion, ko_results = simulate_knockout(qualified, features_dict, model)

    return champion, all_matches, qualified_by_group, ko_results



# MONTE CARLO

def simulate_monte_carlo(groups, features_dict, model, n_simulations=1000):
    results = {}

    for _ in range(n_simulations):
        champion, _, _, _ = simulate_tournament(groups, features_dict, model)
        results[champion] = results.get(champion, 0) + 1

    for team in results:
        results[team] /= n_simulations

    return results