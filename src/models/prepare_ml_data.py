def prepare_ml_data(df):

    df = df.copy()

    y = df["target"]


    # RECONSTRUIR FEATURES DERIVADAS
    # (solo si no vienen ya calculadas)

    if "elo_diff" not in df.columns:
        df["elo_diff"] = df["home_elo"] - df["away_elo"]

    if "form_points_diff" not in df.columns:
        df["form_points_diff"] = (
            df["home_form_points"] - df["away_form_points"]
        )

    if "goal_diff_recent" not in df.columns:
        df["goal_diff_recent"] = (
            (df["home_goals_for"] - df["home_goals_against"]) -
            (df["away_goals_for"] - df["away_goals_against"])
        )


    # FEATURES FINALES (11 columnas)

    features = [
        "home_elo",
        "away_elo",
        "home_form_points",
        "away_form_points",
        "home_goals_for",
        "home_goals_against",
        "away_goals_for",
        "away_goals_against",
        "elo_diff",
        "form_points_diff",
        "goal_diff_recent"
    ]


    # VALIDACIÓN ESTRICTA

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en dataset: {missing}")

    df = df.dropna(subset=features)

    X = df[features]

    # asegurar alineación con índice
    y = y.loc[X.index]

    return X, y