import pickle
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# Carregar o modelo
model_path = 'models/titanic_model.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.get("/")
def read_root():
    return {"message": "API do Titanic funcionando!"}

@app.post("/predict")
def predict(data: dict):
    # Transforme o dicionário de entrada em DataFrame
    df = pd.DataFrame([data])

    # Realize a previsão com o modelo carregado
    prediction = model.predict(df)

    return {"prediction": prediction.tolist()}
