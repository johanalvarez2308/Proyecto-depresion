# src/predict.py
import joblib
import pandas as pd


class DepressionPredictor:

    def __init__(self):
        self.modelo = joblib.load("models/modelo_depresion.joblib")
        self.columnas = joblib.load("models/modelo_columns.joblib")

    def predecir(self, datos_usuario_dict):

        df = pd.DataFrame([datos_usuario_dict])

        # asegurar que coinciden columnas exactas del modelo
        df = df.reindex(columns=self.columnas, fill_value=0)

        pred = self.modelo.predict(df)[0]
        proba = self.modelo.predict_proba(df)[0][1]

        resultado = "DEPRIMIDO" if pred == 1 else "NO DEPRIMIDO"
        return resultado, proba
