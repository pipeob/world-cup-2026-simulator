import pandas as pd

def calculate_team_form(df, window=5):

    df = df.sort_values("date").copy()


    # FORMATO LONG (por equipo)

    home = pd.DataFrame({
        "date": df["date"],
        "team": df["home_team"],
        "goals_for": df["home_score"],
        "goals_against": df["away_score"],
        "target": df["target"]
    })

    away = pd.DataFrame({
        "date": df["date"],
        "team": df["away_team"],
        "goals_for": df["away_score"],
        "goals_against": df["home_score"],
        "target": df["target"].map({2: 0, 1: 1, 0: 2})
    })

    long_df = pd.concat([home, away], ignore_index=True)


    # POINTS

    long_df["points"] = long_df["target"].map({2: 3, 1: 1, 0: 0})

    long_df = long_df.sort_values("date")


    # FORM POR EQUIPO
    
    long_df["form_points"] = (
        long_df.groupby("team")["points"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

    long_df["form_goals_for"] = (
        long_df.groupby("team")["goals_for"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

    long_df["form_goals_against"] = (
        long_df.groupby("team")["goals_against"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )


    # MAPEAR A HOME / AWAY

    # crear clave única
    df["match_id"] = df.index

    home_map = long_df.copy()
    away_map = long_df.copy()

    # HOME
    home_map = home_map.rename(columns={
        "team": "home_team",
        "form_points": "home_form_points",
        "form_goals_for": "home_goals_for",
        "form_goals_against": "home_goals_against"
    })

    # AWAY
    away_map = away_map.rename(columns={
        "team": "away_team",
        "form_points": "away_form_points",
        "form_goals_for": "away_goals_for",
        "form_goals_against": "away_goals_against"
    })

    # merge HOME
    df = df.merge(
        home_map[[
            "date", "home_team",
            "home_form_points",
            "home_goals_for",
            "home_goals_against"
        ]],
        on=["date", "home_team"],
        how="left"
    )

    # merge AWAY
    df = df.merge(
        away_map[[
            "date", "away_team",
            "away_form_points",
            "away_goals_for",
            "away_goals_against"
        ]],
        on=["date", "away_team"],
        how="left"
    )

    return df