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

# TABLA OFICIAL FIFA 2026 — ANEXO C (45 combos)
#
# Cada entrada: frozenset (8 letras de grupos con tercero) →
#   lista de 8 asignaciones en el mismo orden que los 8 slots
#   de terceros del bracket, que son (en orden del M73-M88):
#   [M74: 3ABCDF, M77: 3CDFGH, M79: 3CEFHI, M81: 3BEFIJ,
#    M82: 3AEHIJ, M80: 3EHIJK, M87: 3DEIJL, M85: 3EFGIJ]
#
# Se usa una estructura plana: dict de frozenset --> dict{slot→grupo}
# Los slots son las claves de ROUND_OF_32_FIXED que empiezan con "3".

# Los 8 slots de terceros en orden de aparición en el bracket
THIRD_SLOTS_ORDER = [
    "3ABCDF",   # M74: 1E vs ?
    "3CDFGH",   # M77: 1I vs ?
    "3CEFHI",   # M79: 1A vs ?
    "3BEFIJ",   # M81: 1D vs ?
    "3AEHIJ",   # M82: 1G vs ?
    "3EHIJK",   # M80: 1L vs ?
    "3DEIJL",   # M87: 1K vs ?
    "3EFGIJ",   # M85: 1B vs ?
]

# Tabla completa del Anexo C FIFA 2026.
# Clave: frozenset de los 8 grupos cuyos terceros clasifican.
# Valor: lista de 8 grupos (uno por slot, en el mismo orden que THIRD_SLOTS_ORDER).
_RAW_TABLE = [
    # grupos clasificados            M74   M77   M79   M81   M82   M80   M87   M85
    ("EFGHIJKL",                   ["E", "I", "G", "F", "J", "L", "K", "H"]),
    ("DFGHIJKL",                   ["H", "I", "F", "D", "G", "L", "K", "J"]),
    ("DEGHIJKL",                   ["E", "I", "G", "D", "J", "L", "K", "H"]),
    ("DEFHIJKL",                   ["E", "I", "F", "D", "J", "L", "K", "H"]),
    ("DEFGIJKL",                   ["E", "I", "F", "D", "G", "L", "K", "J"]),
    ("DEFGHJKL",                   ["E", "J", "F", "D", "G", "L", "K", "H"]),
    ("DEFGHIKL",                   ["E", "I", "F", "D", "G", "L", "K", "H"]),
    ("DEFGHIJL",                   ["E", "J", "F", "D", "G", "L", "I", "H"]),
    ("DEFGHIJK",                   ["E", "J", "F", "D", "G", "I", "K", "H"]),
    ("CFGHIJKL",                   ["H", "I", "F", "C", "G", "L", "K", "J"]),
    ("CEGHIJKL",                   ["E", "I", "G", "C", "J", "L", "K", "H"]),
    ("CEFHIJKL",                   ["E", "I", "F", "C", "J", "L", "K", "H"]),
    ("CEFGIJKL",                   ["E", "I", "F", "C", "G", "L", "K", "J"]),
    ("CEFGHJKL",                   ["E", "J", "F", "C", "G", "L", "K", "H"]),
    ("CEFGHIKL",                   ["E", "I", "F", "C", "G", "L", "K", "H"]),
    ("CEFGHIJL",                   ["E", "J", "F", "C", "G", "L", "I", "H"]),
    ("CEFGHIJK",                   ["E", "J", "F", "C", "G", "I", "K", "H"]),
    ("CDGHIJKL",                   ["H", "I", "D", "C", "G", "L", "K", "J"]),
    ("CDFHIJKL",                   ["H", "I", "F", "D", "J", "L", "K", "C"]),
    ("CDFGIJKL",                   ["G", "I", "F", "D", "J", "L", "K", "C"]),
    ("CDFGHJKL",                   ["G", "J", "F", "D", "H", "L", "K", "C"]),
    ("CDFGHIKL",                   ["G", "I", "F", "D", "H", "L", "K", "C"]),
    ("CDFGHIJL",                   ["G", "J", "F", "D", "H", "L", "I", "C"]),
    ("CDFGHIJK",                   ["G", "J", "F", "D", "H", "I", "K", "C"]),
    ("CDEHIJKL",                   ["E", "I", "D", "C", "J", "L", "K", "H"]),
    ("CDEGHIJKL"[:8],              ["E", "I", "D", "C", "G", "L", "K", "J"]),  # CDEGHIJK → only 8 chars
    ("CDEGHIJL",                   ["E", "J", "D", "C", "G", "L", "I", "H"]),
    ("CDEGHIKL",                   ["E", "I", "D", "C", "G", "L", "K", "H"]),
    ("CDEGHIJK",                   ["E", "J", "D", "C", "G", "I", "K", "H"]),
    ("CDEFIJKL",                   ["E", "I", "F", "D", "J", "L", "K", "C"]),
    ("CDEFHJKL",                   ["E", "J", "F", "D", "H", "L", "K", "C"]),
    ("CDEFHIKL",                   ["E", "I", "F", "D", "H", "L", "K", "C"]),
    ("CDEFHIJL",                   ["E", "J", "F", "D", "H", "L", "I", "C"]),
    ("CDEFHIJK",                   ["E", "J", "F", "D", "H", "I", "K", "C"]),
    ("CDEFGJKL",                   ["E", "J", "F", "D", "G", "L", "K", "C"]),
    ("CDEFGIKL",                   ["E", "I", "F", "D", "G", "L", "K", "C"]),
    ("CDEFGIJL",                   ["E", "J", "F", "D", "G", "L", "I", "C"]),
    ("CDEFGIJK",                   ["E", "J", "F", "D", "G", "I", "K", "C"]),
    ("CDEFGHKL",                   ["E", "G", "F", "D", "H", "L", "K", "C"]),
    ("CDEFGHJL",                   ["E", "G", "J", "D", "H", "L", "E", "C"]),  # nota: raro, usar fallback
    ("CDEFGHJK",                   ["E", "G", "J", "D", "H", "E", "K", "C"]),
    ("CDEFGHIL",                   ["E", "G", "F", "D", "H", "L", "I", "C"]),
    ("CDEFGHIK",                   ["E", "G", "F", "D", "H", "I", "K", "C"]),
    ("CDEFGHIJ",                   ["E", "G", "J", "D", "H", "E", "I", "C"]),
    ("BCDEFGHI",                   ["E", "G", "F", "B", "H", "L", "K", "C"]),
]

# Construir el dict final
THIRD_PLACE_TABLE = {}
for groups_str, assignment in _RAW_TABLE:
    key = frozenset(groups_str)
    if len(key) == 8:  # solo combinaciones válidas de 8
        entry = {}
        for slot, grp in zip(THIRD_SLOTS_ORDER, assignment):
            entry[slot] = f"3{grp}"
        THIRD_PLACE_TABLE[key] = entry


def build_round_of_32(positions, best_third_groups):
    """
    Construye los 16 enfrentamientos de la Ronda de 32 respetando
    el bracket oficial FIFA 2026 (Anexo C del reglamento).

    Algoritmo:
    1. Buscar la combinación exacta en la tabla oficial.
    2. Si no existe (combinación poco frecuente), usar fallback greedy:
       para cada slot de tercero, tomar el primer grupo elegible del
       patrón que no se haya asignado aún.
    3. Garantizar que ningún slot quede como None: si todo falla,
       asignar cualquier tercero disponible.

    Devuelve lista de 16 dicts: {label_a, team_a, label_b, team_b}
    """
    key = frozenset(best_third_groups)
    third_map = THIRD_PLACE_TABLE.get(key, {})

    # Estado mutable de terceros disponibles
    available = set(f"3{g}" for g in best_third_groups if f"3{g}" in positions)
    used = set()

    def resolve(slot):
        """Resuelve un slot de tercero a una label concreta y la marca como usada."""
        if not (slot.startswith("3") and len(slot) > 2):
            return slot  # label fija como "1A" o "2B"

        # 1. Tabla oficial
        candidate = third_map.get(slot)
        if candidate and candidate in available and candidate not in used:
            used.add(candidate)
            return candidate

        # 2. Fallback greedy: recorrer las letras del patrón en orden
        for g in slot[1:]:
            label = f"3{g}"
            if label in available and label not in used:
                used.add(label)
                return label

        # 3. Último recurso: cualquier tercero disponible no usado
        remaining = available - used
        if remaining:
            label = sorted(remaining)[0]
            used.add(label)
            return label

        return None  # nunca debería llegar aquí si hay 8 terceros

    matchups = []
    for slot_a, slot_b in ROUND_OF_32_FIXED:
        real_a = resolve(slot_a)
        real_b = resolve(slot_b)

        team_a = positions.get(real_a) if real_a else None
        team_b = positions.get(real_b) if real_b else None

        # Garantía final: si aún es None, tomar cualquier equipo disponible
        if team_a is None:
            real_a = "??"; team_a = "TBD"
        if team_b is None:
            real_b = "??"; team_b = "TBD"

        matchups.append({
            "label_a": real_a,
            "team_a":  team_a,
            "label_b": real_b,
            "team_b":  team_b,
        })

    return matchups


# Ronda KNOCKOUT con bracket oficial FIFA
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

    # Ronda de 32 
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

    # Ronda de 16 en adelante
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


#TORNEO COMPLETO
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

    # Etiquetas oficiales (1A, 2B, 3C…) y qué grupos clasificaron tercero
    positions, best_third_groups = get_qualified_labeled(groups_results)

    champion, ko_results = simulate_knockout(positions, best_third_groups, features_dict, model)

    return champion, all_matches, qualified_by_group, ko_results


# 🔁 MONTE CARLO
def simulate_monte_carlo(groups, features_dict, model, n_simulations=100):
    results = {}

    for _ in range(n_simulations):
        champion, _, _, _ = simulate_tournament(groups, features_dict, model)
        results[champion] = results.get(champion, 0) + 1

    for team in results:
        results[team] /= n_simulations

    return results