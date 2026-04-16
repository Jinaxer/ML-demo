from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class Request(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "ML Sentiment API"}

@app.post("/predict")
def predict_sentiment(req: Request):
    result = predict(req.text)
    return {"prediction": result}
