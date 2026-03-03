import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def summarize_clusters(input_file, output_file):

    print("Loading clustered data...")
    data = pd.read_csv(input_file)

    grouped = data.groupby("cluster")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading FLAN-T5-BASE model...")
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    cluster_summaries = []

    for cluster_id, group in grouped:

        print(f"Processing cluster {cluster_id}")

        count = len(group)

        # Take representative messages
        sample_messages = group["message"].astype(str).head(5).tolist()

        # Remove timestamps (basic cleaning)
        cleaned_messages = []
        for msg in sample_messages:
            if "status=" in msg:
                msg = msg.split("status=")[-1]
            cleaned_messages.append(msg)

        combined_text = " ".join(cleaned_messages)

        # Skip queue activity
        if "queue active" in combined_text.lower():
            summary = "Mail queue processing activity detected (normal system behavior)."

        else:
            prompt = f"""
You are a senior email infrastructure engineer.

The following mail server errors occurred {count} times:

{combined_text}

Explain the most likely root cause in ONE clear technical sentence.
"""

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        cluster_summaries.append({
            "cluster": cluster_id,
            "count": count,
            "root_cause": summary
        })

    output_df = pd.DataFrame(cluster_summaries)
    output_df.to_csv(output_file, index=False)

    print("Root cause report generated successfully.")


if __name__ == "__main__":
    summarize_clusters(
        "outputs/valid/root_cause_results.csv",
        "outputs/valid/final_cluster_summary.csv"
    )