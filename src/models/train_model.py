import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from collections import Counter

from .prepare_ml_data import prepare_ml_data

# Carga y preparación de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "../../data/processed/model_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

# Orden cronológico (para evitar data leakage)
df = df.sort_values("date")

# Split temporal
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Features y target
X_train, y_train = prepare_ml_data(train_df)
X_test, y_test = prepare_ml_data(test_df)


# Manejo de desbalance de clases

counter = Counter(y_train)
total = sum(counter.values())

class_weights = {
    cls: total / count for cls, count in counter.items()
}

sample_weights = y_train.map(class_weights)


# XGBoost

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)


# Predicciones

y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)


# TOP-2 LOGIC 

# Estrategia:
# - Se ordenan las probabilidades
# - Si el empate está entre las 2 mejores opciones
# - Y es suficientemente cercano a la mejor → se predice empate

CLOSE_DIFF = 0.03  # qué tan cerca debe estar del top 1

y_pred_custom = []

for p in probs:
    sorted_idx = np.argsort(p)[::-1]  # índices ordenados de mayor a menor
    
    top1 = sorted_idx[0]
    top2 = sorted_idx[1]

    # Si empate está en el top 2 y es competitivo
    if 1 in [top1, top2] and abs(p[1] - p[top1]) < CLOSE_DIFF:
        y_pred_custom.append(1)
    else:
        y_pred_custom.append(top1)

#print(X_train.columns.tolist())

# guardar modelo
joblib.dump(model, "trained_models/modelo_pipeline.pkl")

# guardar columnas (MUY importante)
joblib.dump(X_train.columns.tolist(), "trained_models/columns.pkl")

importances = model.feature_importances_

feature_names = model.get_booster().feature_names

'''df_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

df_importance = df_importance.sort_values(by="importance", ascending=False)

print(df_importance)'''