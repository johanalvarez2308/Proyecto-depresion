# main.py
from src.data_loader import DataLoader
from src.train_model import DepressionModel

def main():
    print("Entrenando modelo de depresión...\n")

    loader = DataLoader("data/raw/student_depression_dataset.csv")
    df = loader.cargar_datos()

    modelo = DepressionModel()
    metricas = modelo.entrenar(df)

    print("Entrenamiento completado.")
    print(metricas)

if __name__ == "__main__":
    main()
