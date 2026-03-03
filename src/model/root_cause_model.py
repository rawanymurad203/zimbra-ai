import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os


def find_best_k(X):
    sample_size = 10000

    if X.shape[0] > sample_size:
        print("sampling data for speed...")
        X_sample = X[:sample_size]
    else:
        X_sample = X

    best_k = 2
    best_score = -1

    for k in range(2, 8):
        print(f"testing k={k}")
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_sample)

        score = silhouette_score(X_sample, labels)

        print(f"score={score}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"best number of clusters: {best_k}")
    return best_k

def train_root_cause_model(input_file):
    print("loading data...")

    data = pd.read_csv(input_file)
    messages = data["message"].astype(str)

    print("vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(messages)

    print("finding best number of clusters...")
    best_k = find_best_k(X)

    print("training clustering model...")
    model = KMeans(n_clusters=best_k, random_state=42)
    model.fit(X)

    print("saving model...")
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/root_cause_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("done")


if __name__ == "__main__":
    train_root_cause_model("outputs/train/parsed_logs.csv")