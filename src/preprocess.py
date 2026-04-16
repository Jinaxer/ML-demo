import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    df = pd.read_csv(path)
    return df['text'], df['label']

def build_vectorizer():
    return TfidfVectorizer(stop_words='english')

def transform(vectorizer, texts):
    return vectorizer.transform(texts)
