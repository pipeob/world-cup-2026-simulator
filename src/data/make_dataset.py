import pandas as pd
from pathlib import Path
from .clean_data import load_and_clean_data, add_target_variable
from src.features.form_features import calculate_team_form
from src.features.build_features import create_diff_features
from src.features.elo_features import calculate_elo


def create_model_dataset(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Cargar y limpiar
    df = load_and_clean_data(input_path)

    # Crear target
    df = add_target_variable(df)

    # Crear dataset simétrico
    home_df = df.copy()
    away_df = df.copy()

    # Invertir equipos
    away_df["home_team"], away_df["away_team"] = df["away_team"], df["home_team"]
    away_df["home_score"], away_df["away_score"] = df["away_score"], df["home_score"]

    # Recalcular target
    away_df["target"] = away_df["target"].map({2: 0, 1: 1, 0: 2})

    # Unir
    df_model = pd.concat([home_df, away_df], ignore_index=True)

    # Ordenar por fecha
    df_model["date"] = pd.to_datetime(df_model["date"])
    df_model = df_model.sort_values("date")

    # Calculo forma reciente, diferencia de puntos y goles, y Elo
    df_model = calculate_team_form(df_model)
    df_model = calculate_elo(df_model)
    df_model = create_diff_features(df_model)

    print("Columnas después de form_features:")
    print(df_model.columns)

    # Guarda el modelo
    df_model.to_csv(output_path, index=False)

    print("Dataset creado correctamente")
    print(f"Filas totales: {len(df_model)}")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[2]
    create_model_dataset(
        input_path=root_dir / "data" / "raw" / "results.csv",
        output_path=root_dir / "data" / "processed" / "model_dataset.csv"
    )