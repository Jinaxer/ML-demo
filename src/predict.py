import pickle

MODEL_PATH = "model/model.pkl"
VEC_PATH = "model/vectorizer.pkl"

def load():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

def predict(text):
    model, vectorizer = load()
    X = vectorizer.transform([text])
    return int(model.predict(X)[0])
