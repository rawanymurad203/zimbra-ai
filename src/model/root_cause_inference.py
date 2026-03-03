import pandas as pd
import joblib


def run_inference(input_file, output_file):
    print("loading data...")

    # FIX: handle bad CSV formatting (commas inside text)
    data = pd.read_csv(
        input_file,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip"
    )

    print("loading model...")

    model = joblib.load("models/root_cause_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    print("vectorizing...")

    X = vectorizer.transform(data["message"].astype(str))

    print("predicting clusters...")

    clusters = model.predict(X)

    data["cluster"] = clusters

    print("saving results...")

    data.to_csv(output_file, index=False)

    print("done")


if __name__ == "__main__":
    run_inference(
        "outputs/valid/parsed_logs.csv",
        "outputs/valid/root_cause_results.csv"
    )