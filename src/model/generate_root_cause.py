import pandas as pd
from transformers import pipeline


def generate_root_causes(input_file, output_file):
    print("loading data...")
    data = pd.read_csv(input_file)

    print("loading text generation model...")
    generator = pipeline(
        "text-generation",   # correct task
        model="google/flan-t5-small"
    )

    root_causes = []

    print("generating root causes...")

    for i, row in data.iterrows():
        log_text = str(row["message"])

        # create prompt
        prompt = f"Summarize the main issue in these mail logs in ONE short sentence: {log_text}"

        # generate output
        result = generator(
    prompt,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.2
)
        text = result[0]["generated_text"]

        # remove prompt if repeated
        cleaned = text.replace(prompt, "").strip()

        # keep only first sentence
        description = cleaned.split(".")[0].strip()

        # fallback if model output is weak
        if len(description) < 10:
            description = "Mail delivery issue detected"

        root_causes.append(description)

        if i % 50 == 0:
            print(f"processed {i} rows")

    # save results
    data["root_cause"] = root_causes
    data.to_csv(output_file, index=False)

    print("done")


if __name__ == "__main__":
    generate_root_causes(
        input_file="outputs/valid/parsed_logs.csv",
        output_file="outputs/valid/final_with_root_cause.csv"
    )