# app.py
import streamlit as st
from src.predict import DepressionPredictor


st.title("😔 Predicción de Depresión en Estudiantes")
st.write("Ingrese los datos para evaluar riesgo de depresión.")

# Campos de entrada
age = st.number_input("Edad", min_value=10, max_value=80, value=20)
academic = st.number_input("Academic Pressure", 0, 10, 1)
study_satisfaction = st.number_input("Study Satisfaction", 0, 10, 1)
work_hours = st.number_input("Work/Study Hours", 0, 24, 2)
financial = st.number_input("Financial Stress", 0, 10, 3)

gender = st.selectbox("Género", ["Male", "Female"])
diet = st.selectbox("Hábitos Alimenticios", 
                    ["Healthy", "Moderate", "Unhealthy", "Others"])
suicide = st.selectbox("¿Ha tenido pensamientos suicidas?", ["Yes", "No"])
sleep = st.selectbox("Duración del sueño", [
    "Less than 5 hours",
    "5-6 hours",
    "7-8 hours",
    "More than 8 hours",
    "Others"
])
family_history = st.selectbox("Historial familiar de enfermedad mental", ["Yes", "No"])


# Mapeo para crear dummies iguales a entrenamiento
entrada = {
    "Age": age,
    "Academic Pressure": academic,
    "Study Satisfaction": study_satisfaction,
    "Work/Study Hours": work_hours,
    "Financial Stress": financial,

    # dummies manuales
    f"Gender_Male": 1 if gender == "Male" else 0,
    f"Gender_Female": 1 if gender == "Female" else 0,

    f"Diet_{diet}": 1,

    f"Suicidal_{suicide}": 1,

    f"Sleep_{sleep}": 1,

    f"Family_History_{family_history}": 1
}


if st.button("🔍 Predecir Depresión"):

    predictor = DepressionPredictor()
    resultado, prob = predictor.predecir(entrada)

    st.subheader("📊 Resultado")
    st.write(f"Estado: **{resultado}**")
    st.write(f"Probabilidad: **{prob*100:.2f}%**")

