import os
import json
import math
from src.matcher import solve

SAMPLES_DIR = "samples"
NUM_SAMPLES = 50


def evaluate():
    errors = []
    print("APEX DRIFT-SENSE EXTENDED EVALUATION")
    print("====================================")

    for i in range(NUM_SAMPLES):
        search_path = os.path.join(SAMPLES_DIR, f"search_{i:04d}.png")
        reference_path = os.path.join(SAMPLES_DIR, f"reference_{i:04d}.png")
        metadata_path = os.path.join(SAMPLES_DIR, f"metadata_{i:04d}.json")

        if not os.path.exists(search_path):
            continue

        with open(metadata_path, "r") as f:
            gt = json.load(f)["ground_truth"]

        result, candidates = solve(search_path, reference_path)
        if result is None:
            continue

        error = math.hypot(result["x"] - gt["x"], result["y"] - gt["y"])
        errors.append(error)

    if not errors:
        print("No successful predictions.")
        return

    errors.sort()
    mean_error = sum(errors) / len(errors)
    median_error = errors[len(errors) // 2]

    print(f"Successful samples : {len(errors)}/{NUM_SAMPLES}")
    print(f"Mean pixel error   : {mean_error:.2f}px")
    print(f"Median pixel error : {median_error:.2f}px")
    print(f"Within 5 px        : {sum(e <= 5 for e in errors)/len(errors)*100:.2f}%")
    print(f"Within 10 px       : {sum(e <= 10 for e in errors)/len(errors)*100:.2f}%")


if __name__ == "__main__":
    evaluate()
