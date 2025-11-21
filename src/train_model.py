# src/train_model.py
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.preprocessing import Preprocesador


class DepressionModel:

    def entrenar(self, df):

        pre = Preprocesador()
        X, y = pre.preparar_datos(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        modelo = LogisticRegression()
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred), 3),
            "recall": round(recall_score(y_test, y_pred), 3),
            "f1": round(f1_score(y_test, y_pred), 3),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 3)
        }

        # Ensure output directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        joblib.dump(modelo, "models/modelo_depresion.joblib")
        joblib.dump(list(X.columns), "models/modelo_columns.joblib")

        with open("reports/metricas.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics
