import pickle
from sklearn.linear_model import LogisticRegression
from preprocess import load_data, build_vectorizer

DATA_PATH = "data/sample_data.csv"
MODEL_PATH = "model/model.pkl"
VEC_PATH = "model/vectorizer.pkl"

def train():
    X_text, y = load_data(DATA_PATH)

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(X_text)

    model = LogisticRegression()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model trained and saved.")

if __name__ == "__main__":
    train()
