import pandas as pd
import numpy as np

def build_team_features(df):

    df = df.sort_values("date")

    team_features = {}

    teams = pd.concat([df["home_team"], df["away_team"]]).unique()

    for team in teams:

        home_games = df[df["home_team"] == team]
        away_games = df[df["away_team"] == team]

        last_home = home_games.tail(1)
        last_away = away_games.tail(1)

        if not last_home.empty:
            row = last_home.iloc[0]

            features = {
                "elo": row["home_elo"],
                "form_points": row["form_points"],
                "goals_for": row["form_goals_for"],
                "goals_against": row["form_goals_against"]
            }

        elif not last_away.empty:
            row = last_away.iloc[0]

            features = {
                "elo": row["away_elo"],
                "form_points": row["form_points"],
                "goals_for": row["form_goals_for"],
                "goals_against": row["form_goals_against"]
            }

        else:
            continue

        team_features[team] = features

    return team_features

def build_match_features(team_a, team_b, team_features, columns):

    a = team_features[team_a]
    b = team_features[team_b]

    data = {
        "home_elo": a["elo"],
        "away_elo": b["elo"],
        "elo_diff": a["elo"] - b["elo"],

        "form_points": a["form_points"],
        "form_goals_for": a["goals_for"],
        "form_goals_against": a["goals_against"],

        "form_points_diff": a["form_points"] - b["form_points"],
        "goal_diff_recent": (a["goals_for"] - a["goals_against"]) - (b["goals_for"] - b["goals_against"]),
    }

    # ORDEN EXACTO DEL MODELO
    return np.array([data[col] for col in columns])