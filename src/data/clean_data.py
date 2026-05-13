import pandas as pd


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Convierte fecha correctamente
    df["date"] = pd.to_datetime(df["date"])

    # Filtro fútbol moderno
    df = df[df["date"] >= "2000-01-01"]

    # Elimina filas sin marcador
    df = df.dropna(subset=["home_score", "away_score"])

    return df


def add_target_variable(df):
    def get_result(row):
        if row["home_score"] > row["away_score"]:
            return "home_win"
        elif row["home_score"] < row["away_score"]:
            return "away_win"
        else:
            return "draw"

    df["result"] = df.apply(get_result, axis=1)

    mapping = {"home_win": 2, "draw": 1, "away_win": 0}
    df["target"] = df["result"].map(mapping)

    return df