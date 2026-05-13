def calculate_elo(df, k=20):

    df = df.sort_values("date")

    # Inicializar ratings
    elo_dict = {}

    home_elo = []
    away_elo = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # valor inicial
        if home not in elo_dict:
            elo_dict[home] = 1500
        if away not in elo_dict:
            elo_dict[away] = 1500

        h_elo = elo_dict[home]
        a_elo = elo_dict[away]

        home_elo.append(h_elo)
        away_elo.append(a_elo)

        # probabilidad esperada
        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))

        # resultado real
        if row["target"] == 2:
            score = 1
        elif row["target"] == 1:
            score = 0.5
        else:
            score = 0

        # actualizar
        elo_dict[home] = h_elo + k * (score - expected_home)
        elo_dict[away] = a_elo + k * ((1 - score) - (1 - expected_home))

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo

    return df