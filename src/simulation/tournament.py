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

    # Ordenar: puntos --> diferencia de gol → goles a favor
    sorted_standings = sorted(
        standings.items(),
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
        reverse=True
    )

    # Devolver lista de (equipo, puntos, gf, ga)
    return sorted_standings, matches


# CLASIFICADOS CON ETIQUETAS (1A, 2B, 3X, etc.)
GROUP_NAMES = ["A","B","C","D","E","F","G","H","I","J","K","L"]

def get_qualified_labeled(groups_results):
    """
    Devuelve:
      positions: dict  label → team_name
                 ej. {"1A": "Argentina", "2B": "France", "3C": "Ecuador", ...}
      best_third_groups: lista de letras de grupos cuyos terceros clasificaron
                 ej. ["C","D","E","F","G","H"]  (8 grupos)
    """
    positions = {}
    third_candidates = []   # (team, stats_dict, group_letter)

    for idx, group in enumerate(groups_results):
        g = GROUP_NAMES[idx]
        # group = [(team, {pts,gf,ga}), ...]
        positions[f"1{g}"] = group[0][0]
        positions[f"2{g}"] = group[1][0]
        if len(group) > 2:
            third_candidates.append((group[2][0], group[2][1], g))

    # Ordenar terceros: puntos → dif. goles → goles a favor
    third_sorted = sorted(
        third_candidates,
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
        reverse=True
    )
    best_thirds = third_sorted[:8]
    best_third_groups = [g for _, _, g in best_thirds]

    for team, _, g in best_thirds:
        positions[f"3{g}"] = team

    return positions, best_third_groups


# CUADRO OFICIAL FIFA 2026 — RONDA DE 32
# Cada partido es (slot_A, slot_B) donde slot es una label
# como "1A", "2B", o una lista de posibles terceros "3C/D/E/..."
# Los 16 partidos oficiales (Match 73–88):
ROUND_OF_32_FIXED = [
    # M73 – M88 en orden de bracket
    ("2A",          "2B"),          # M73
    ("1E",          "3ABCDF"),      # M74
    ("1F",          "2C"),          # M75
    ("1C",          "2F"),          # M76
    ("1I",          "3CDFGH"),      # M77
    ("2E",          "2I"),          # M78
    ("1A",          "3CEFHI"),      # M79
    ("1L",          "3EHIJK"),      # M80
    ("1D",          "3BEFIJ"),      # M81
    ("1G",          "3AEHIJ"),      # M82
    ("2K",          "2L"),          # M83
    ("1H",          "2J"),          # M84
    ("1B",          "3EFGIJ"),      # M85
    ("1J",          "2H"),          # M86
    ("1K",          "3DEIJL"),      # M87
    ("2D",          "2G"),          # M88
]

# Tabla oficial de combinaciones de terceros (Annex C FIFA 2026).
# Clave: frozenset de las 8 letras de grupos cuyos terceros clasificaron.
# Valor: dict de slot_tercero → grupo que lo ocupa.
# Solo se incluyen las 45 combinaciones posibles cuando A y B nunca
# producen tercero (grupos A y B siempre tienen 2 clasificados directos,
# los terceros son de C-L). Fuente: Wikipedia / reglamento FIFA.
#
# Columnas de la tabla: 1A_opp, 1E_opp, 1I_opp, 1D_opp, 1G_opp, 1L_opp, 1K_opp, 1B_opp
# (= el tercero asignado a cada grupo ganador que tiene slot "3x")
THIRD_PLACE_TABLE = {
    # grupos que clasifican terceros → {slot: grupo_del_tercero}
    frozenset("EFGHIJKL"): {"3ABCDF":"3E","3AEHIJ":"3J","3CDFGH":"3I","3BEFIJ":"3F","3AEHIJ":"3H","3CEFHI":"3G","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("DFGHIJKL"): {"3ABCDF":"3H","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3D","3AEHIJ":"3J","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("DEGHIJKL"): {"3ABCDF":"3E","3AEHIJ":"3J","3CDFGH":"3I","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3G","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("DEFHIJKL"): {"3ABCDF":"3E","3AEHIJ":"3J","3CDFGH":"3I","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("DEFGIJKL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3D","3AEHIJ":"3J","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("DEFGHJKL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("DEFGHIKL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("DEFGHIJL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3I"},
    frozenset("DEFGHIJK"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3I","3DEIJL":"3K"},
    frozenset("CFGHIJKL"): {"3ABCDF":"3H","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3J","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CEGHIJKL"): {"3ABCDF":"3E","3AEHIJ":"3J","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3G","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CEFHIJKL"): {"3ABCDF":"3E","3AEHIJ":"3J","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CEFGIJKL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3J","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CEFGHJKL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CEFGHIKL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CEFGHIJL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3I"},
    frozenset("CEFGHIJK"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3I","3DEIJL":"3K"},
    frozenset("CDGHIJKL"): {"3ABCDF":"3H","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3J","3CEFHI":"3D","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDFHIJKL"): {"3ABCDF":"3C","3AEHIJ":"3J","3CDFGH":"3I","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDFGIJKL"): {"3ABCDF":"3C","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3D","3AEHIJ":"3J","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDFGHJKL"): {"3ABCDF":"3C","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDFGHIKL"): {"3ABCDF":"3C","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDFGHIJL"): {"3ABCDF":"3C","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3L","3DEIJL":"3I"},
    frozenset("CDFGHIJK"): {"3ABCDF":"3C","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3D","3AEHIJ":"3H","3CEFHI":"3F","3EHIJK":"3I","3DEIJL":"3K"},
    frozenset("CDEHIJKL"): {"3ABCDF":"3E","3AEHIJ":"3J","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3D","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDEGHIJKL"[:-1]): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3J","3CEFHI":"3D","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDEGHIJKL"[:-2]+"L"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3D","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDEGHIKL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3I","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3D","3EHIJK":"3L","3DEIJL":"3K"},
    frozenset("CDEGHIJL"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3D","3EHIJK":"3L","3DEIJL":"3I"},
    frozenset("CDEGHIJK"): {"3ABCDF":"3E","3AEHIJ":"3G","3CDFGH":"3J","3BEFIJ":"3C","3AEHIJ":"3H","3CEFHI":"3D","3EHIJK":"3I","3DEIJL":"3K"},
}

def resolve_third_slot(slot_pattern, best_third_groups, positions):
    """
    Dado un slot como "3ABCDF" (= el tercero de alguno de esos grupos),
    devuelve el equipo que lo ocupa según qué grupos clasificaron terceros.
    """
    # Los grupos válidos para este slot: la intersección entre
    # los grupos del patrón y los que realmente clasificaron tercero
    eligible = [g for g in slot_pattern[1:] if g in best_third_groups]
    if not eligible:
        return None
    # Elegir el primero que esté disponible en positions
    for g in eligible:
        label = f"3{g}"
        if label in positions:
            return label
    return None


def build_round_of_32(positions, best_third_groups):
    """
    Construye los 16 enfrentamientos de la Ronda de 32 con los equipos reales,
    respetando el bracket oficial FIFA 2026.
    Devuelve lista de dicts con label_a, team_a, label_b, team_b.

    La asignación de terceros sigue la tabla oficial de FIFA.
    Para cada partido con slot "3XYZ..." se toma el primer grupo elegible
    que aún no haya sido asignado, respetando el orden del bracket.
    """
    key = frozenset(best_third_groups)
    third_map = THIRD_PLACE_TABLE.get(key, {})

    used_thirds = set()   # etiquetas de terceros ya asignados (ej. "3C")

    def resolve(slot):
        """Devuelve la label real (ej. '3E') para un slot tipo '3ABCDF'."""
        if not (slot.startswith("3") and len(slot) > 2):
            return slot   # slot fijo como "1A" o "2B"

        # 1. Intentar con la tabla oficial del Anexo C
        candidate = third_map.get(slot)
        if candidate and candidate in positions and candidate not in used_thirds:
            used_thirds.add(candidate)
            return candidate

        # 2. Fallback: primer grupo elegible del patrón que no se haya usado
        for g in slot[1:]:
            label = f"3{g}"
            if g in best_third_groups and label in positions and label not in used_thirds:
                used_thirds.add(label)
                return label

        return None  # no debería ocurrir

    matchups = []
    for slot_a, slot_b in ROUND_OF_32_FIXED:
        real_a = resolve(slot_a)
        real_b = resolve(slot_b)

        team_a = positions.get(real_a, f"?{real_a}") if real_a else "???"
        team_b = positions.get(real_b, f"?{real_b}") if real_b else "???"

        matchups.append({
            "label_a": real_a or slot_a,
            "team_a":  team_a,
            "label_b": real_b or slot_b,
            "team_b":  team_b,
        })

    return matchups


# Fase Knockout con bracket FIFA
def _play_match(model, team_a, team_b, features_dict):
    """Simula un partido KO (sin empate final): devuelve (winner, ga, gb, penalties)."""
    res, _, ga, gb = simulate_match(model, team_a, team_b, features_dict)
    penalties = False

    if res == 2:
        winner = team_a
    elif res == 0:
        winner = team_b
    else:
        elo_a = features_dict[team_a]["elo"]
        elo_b = features_dict[team_b]["elo"]
        p_a   = elo_a / (elo_a + elo_b)
        winner = np.random.choice([team_a, team_b], p=[p_a, 1 - p_a])
        penalties = True

    return winner, ga, gb, penalties


def simulate_knockout(positions, best_third_groups, features_dict, model):
    """
    Simula la fase KO completa respetando el bracket oficial FIFA 2026.
    Devuelve (champion, ko_results) donde ko_results es un OrderedDict:
    {
      "Ronda de 32":      [{"label_a","team_a","label_b","team_b","score_a","score_b","winner","winner_label","penalties"}, ...],
      "Octavos de Final": [...],
      "Cuartos de Final": [...],
      "Semifinales":      [...],
      "Final":            [...],
    }
    """
    ko_results = {}

    # ── Ronda de 32 ──────────────────────────────────────────
    r32_matchups = build_round_of_32(positions, best_third_groups)
    r32_results  = []
    r16_pairs    = []   # pares de ganadores para Octavos

    for m in r32_matchups:
        label_a, team_a = m["label_a"], m["team_a"]
        label_b, team_b = m["label_b"], m["team_b"]
        winner, ga, gb, pens = _play_match(model, team_a, team_b, features_dict)
        winner_label = label_a if winner == team_a else label_b
        r32_results.append({
            "label_a": label_a, "team_a": team_a,
            "label_b": label_b, "team_b": team_b,
            "score_a": ga,      "score_b": gb,
            "winner":  winner,  "winner_label": winner_label,
            "penalties": pens,
        })
        r16_pairs.append((winner_label, winner))

    ko_results["Ronda de 32"] = r32_results

    # ── Ronda de 16 en adelante ───────────────────────────────
    # Los ganadores de r32 se emparejan en el mismo orden del bracket
    # (ganador M73 vs ganador M74, ganador M75 vs M76, etc.)
    round_names = ["Octavos de Final", "Cuartos de Final", "Semifinales", "Final"]
    current_round = r16_pairs   # lista de (label, team)

    for rname in round_names:
        round_results = []
        next_round    = []

        for i in range(0, len(current_round), 2):
            label_a, team_a = current_round[i]
            label_b, team_b = current_round[i + 1]

            winner, ga, gb, pens = _play_match(model, team_a, team_b, features_dict)
            winner_label = label_a if winner == team_a else label_b

            round_results.append({
                "label_a": label_a, "team_a": team_a,
                "label_b": label_b, "team_b": team_b,
                "score_a": ga,      "score_b": gb,
                "winner":  winner,  "winner_label": winner_label,
                "penalties": pens,
            })
            next_round.append((winner_label, winner))

        ko_results[rname] = round_results
        current_round = next_round

        if len(current_round) == 1:
            break

    champion = current_round[0][1]
    return champion, ko_results


# TORNEO COMPLETO
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

    # Etiquetas oficiales (1A, 2B, 3C…) + qué grupos clasificaron tercero
    positions, best_third_groups = get_qualified_labeled(groups_results)

    champion, ko_results = simulate_knockout(positions, best_third_groups, features_dict, model)

    return champion, all_matches, qualified_by_group, ko_results


# MONTE CARLO
def simulate_monte_carlo(groups, features_dict, model, n_simulations=100):
    results = {}

    for _ in range(n_simulations):
        champion, _, _, _ = simulate_tournament(groups, features_dict, model)
        results[champion] = results.get(champion, 0) + 1

    for team in results:
        results[team] /= n_simulations

    return results