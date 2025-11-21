# src/preprocessing.py
import pandas as pd
import numpy as np

class Preprocesador:

    def preparar_datos(self, df):

        # =========================
        # 1. Eliminar columnas
        # =========================
        df = df.drop([
            'id', 'City', 'Profession', 'Work Pressure',
            'CGPA', 'Job Satisfaction', 'Degree'
        ], axis=1)

        # =========================
        # 2. Dummies para variables categóricas
        # =========================
        df['Sleep Duration'] = df['Sleep Duration'].str.replace("'", "")

        gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender', dtype=int)
        fam_hist_dummies = pd.get_dummies(df['Family History of Mental Illness'], 
                                          prefix='Family_History', dtype=int)
        diet_dummies = pd.get_dummies(df['Dietary Habits'], prefix='Diet', dtype=int)
        suicide_dummies = pd.get_dummies(df['Have you ever had suicidal thoughts ?'],
                                         prefix='Suicidal', dtype=int)
        sleep_dummies = pd.get_dummies(df['Sleep Duration'], prefix='Sleep', dtype=int)

        # =========================
        # 3. Construir dataset final
        # =========================
        df = df.drop([
            'Gender', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
            'Sleep Duration', 'Family History of Mental Illness'
        ], axis=1)

        df = pd.concat([df,
                        gender_dummies,
                        fam_hist_dummies,
                        diet_dummies,
                        suicide_dummies,
                        sleep_dummies], axis=1)

        # =========================
        # 4. Limpiar Financial Stress
        # =========================
        df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)
        df['Financial Stress'] = pd.to_numeric(df['Financial Stress'])
        df['Financial Stress'] = df['Financial Stress'].fillna(df['Financial Stress'].mean())

        # =========================
        # 5. Separar X e Y
        # =========================
        y = df['Depression']
        X = df.drop('Depression', axis=1)

        return X, y
