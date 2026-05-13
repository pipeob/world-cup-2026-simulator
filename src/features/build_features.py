def create_diff_features(df):

    df = df.copy()


    df["form_points_diff"] = (
        df["home_form_points"] - df["away_form_points"]
    )

    df["goal_diff_recent"] = (
        (df["home_goals_for"] - df["home_goals_against"]) -
        (df["away_goals_for"] - df["away_goals_against"])
    )

    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df