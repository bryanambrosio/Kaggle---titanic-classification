from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API do Titanic funcionando!"}

@app.get("/predict")
def predict():
    return {"prediction": "Aqui ficará a previsão do modelo!"}
