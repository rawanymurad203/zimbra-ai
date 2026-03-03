import os

from src.preprocessing.parse_logs import parse_logs
from src.model.root_cause_inference import run_inference
from src.model.map_root_cause import map_root_cause


def run_pipeline(input_log, output_prefix):
    os.makedirs(output_prefix, exist_ok=True)

    parsed_file = f"{output_prefix}/parsed_logs.csv"
    cluster_file = f"{output_prefix}/root_cause_results.csv"
    final_file = f"{output_prefix}/root_cause_final.csv"

    print("parsing logs...")
    parse_logs(input_log, parsed_file)

    print("running AI inference...")
    run_inference(parsed_file, cluster_file)

    print("mapping root causes...")
    map_root_cause(cluster_file, final_file)

    print("pipeline done")
    print(f"final output: {final_file}")


if __name__ == "__main__":
    # train run
    run_pipeline(
        "data/split/zimbra_24_train.log",
        "outputs/train"
    )

    # validation run
    run_pipeline(
        "data/split/zimbra_24_valid.log",
        "outputs/valid"
    )