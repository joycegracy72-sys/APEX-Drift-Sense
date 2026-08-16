import os
import json
import cv2
import math

from src.matcher import solve


SAMPLES_DIR = "samples"
NUM_SAMPLES = 50


def calculate_error(predicted, ground_truth):
    dx = predicted["x"] - ground_truth["x"]
    dy = predicted["y"] - ground_truth["y"]

    return math.sqrt(
        dx * dx + dy * dy
    )


def evaluate():

    errors = []
    successful = 0

    print("APEX DRIFT-SENSE EVALUATION")
    print("============================")

    for i in range(NUM_SAMPLES):

        search_path = os.path.join(
            SAMPLES_DIR,
            f"search_{i:04d}.png"
        )

        reference_path = os.path.join(
            SAMPLES_DIR,
            f"reference_{i:04d}.png"
        )

        metadata_path = os.path.join(
            SAMPLES_DIR,
            f"metadata_{i:04d}.json"
        )

        if not os.path.exists(
            search_path
        ):
            continue

        with open(
            metadata_path,
            "r"
        ) as f:

            metadata = json.load(f)

        ground_truth = metadata["ground_truth"]

        result, candidates = solve(
            search_path,
            reference_path
        )

        if result is None:

            print(
                f"[{i:02d}] NO MATCH"
            )

            continue

        error = calculate_error(
            result,
            ground_truth
        )

        errors.append(error)
        successful += 1

        print(
            f"[{i:02d}] "
            f"GT=({ground_truth['x']},{ground_truth['y']}) "
            f"Pred=({result['x']:.1f},{result['y']:.1f}) "
            f"Error={error:.2f}px "
            f"Candidates={len(candidates)}"
        )

    print("\n============================")

    if not errors:

        print("No successful predictions.")

        return

    mean_error = sum(errors) / len(errors)

    median_error = sorted(errors)[
        len(errors) // 2
    ]

    within_5 = sum(
        error <= 5
        for error in errors
    )

    within_10 = sum(
        error <= 10
        for error in errors
    )

    print(
        f"Successful samples : {successful}/{NUM_SAMPLES}"
    )

    print(
        f"Mean pixel error   : {mean_error:.2f}px"
    )

    print(
        f"Median pixel error : {median_error:.2f}px"
    )

    print(
        f"Within 5 px        : "
        f"{within_5 / len(errors) * 100:.2f}%"
    )

    print(
        f"Within 10 px       : "
        f"{within_10 / len(errors) * 100:.2f}%"
    )


if __name__ == "__main__":
    evaluate()