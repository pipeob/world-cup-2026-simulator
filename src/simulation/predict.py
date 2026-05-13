import joblib
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

MODEL_PATH = os.path.join(PROJECT_ROOT, "trained_models/modelo_pipeline.pkl")
COLUMNS_PATH = os.path.join(PROJECT_ROOT, "trained_models/columns.pkl")

model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)


# FEATURES REALES (EJEMPLO)
team_elo = {
    "Colombia": 1800,
    "Brasil": 1900,
    "Argentina": 1950,
    "Uruguay": 1850,
    "Chile": 1750,
    "México": 1780,
    "USA": 1760,
    "España": 1920
}

def build_features(team1, team2):
    data = {
        "elo_diff": team_elo[team1] - team_elo[team2],
        "home_elo": team_elo[team1],
        "away_elo": team_elo[team2],
        "form_points_diff": 0,
        "goal_diff_recent": 0,
        "form_goals_for": 0,
        "form_goals_against": 0,
        "form_points": 0
    }

    df = pd.DataFrame([data])
    return df[columns]


# PREDICCIÓN
def predict_match(team1, team2):
    X = build_features(team1, team2)
    probs = model.predict_proba(X)[0]

    sorted_idx = np.argsort(probs)[::-1]
    top1, top2 = sorted_idx[0], sorted_idx[1]

    CLOSE_DIFF = 0.04

    if abs(probs[top1] - probs[top2]) < CLOSE_DIFF:
        pred = 1
    else:
        pred = top1

    return pred, probs