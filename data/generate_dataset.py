import os
import cv2
import json
import random
import argparse
import numpy as np


IMAGE_SIZE = 1000
TARGET_SIZE = 100
NUM_SAMPLES = 50

OUTPUT_DIR = "samples"

# Supported synthetic architecture styles. These are simple,
# distinguishable visual motifs for benchmarking purposes only - see
# references.md and the limitation notice below. They are NOT physically
# accurate models of any real DRAM or FinFET process.
ARCHITECTURES = ("DRAM", "FINFET")


def create_wafer_pattern(size=IMAGE_SIZE, architecture="DRAM"):
    """
    Synthetic benchmark pattern generator.

    architecture="DRAM": the original repetitive cell-array style
    pattern (regular rectangular cells + interconnect grid), unchanged
    from the original generator.

    architecture="FINFET": the same cell-array/interconnect pipeline,
    with the per-cell motif swapped for a directional, fin-like
    repeated structure (parallel vertical fins + a gate-like crossbar)
    instead of the DRAM rectangle+circle motif. Everything else
    (spacing, interconnect grid, noise, blur) is identical.

    These are synthetic benchmark structures for navigation-error
    recovery testing, NOT claims of physically accurate semiconductor
    process simulation. See references.md.
    """

    architecture = architecture.upper()

    image = np.zeros((size, size), dtype=np.uint8)

    # Repetitive semiconductor-like cells
    cell_w = 40
    cell_h = 40
    spacing = 60

    for y in range(20, size - 40, spacing):

        for x in range(20, size - 40, spacing):

            if architecture == "FINFET":
                # Directional fin-like repeated structure: parallel
                # vertical "fins" instead of a single rectangle outline.
                for fin_x in range(x, x + cell_w, 8):
                    cv2.line(
                        image,
                        (fin_x, y),
                        (fin_x, y + cell_h),
                        150,
                        2
                    )

                # Gate/interconnect-like crossbar over the fins
                cv2.line(
                    image,
                    (x, y + 20),
                    (x + cell_w, y + 20),
                    200,
                    2
                )

                # Small variation
                if random.random() > 0.5:

                    cv2.circle(
                        image,
                        (x + 30, y + 30),
                        4,
                        220,
                        -1
                    )

            else:
                # Main cell
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + cell_w, y + cell_h),
                    150,
                    2
                )

                # Vertical structure
                cv2.line(
                    image,
                    (x + 10, y),
                    (x + 10, y + cell_h),
                    200,
                    1
                )

                # Horizontal structure
                cv2.line(
                    image,
                    (x, y + 20),
                    (x + cell_w, y + 20),
                    200,
                    1
                )

                # Small variation
                if random.random() > 0.5:

                    cv2.circle(
                        image,
                        (x + 30, y + 30),
                        4,
                        220,
                        -1
                    )

    # Long interconnect lines

    for x in range(5, size, 100):

        cv2.line(
            image,
            (x, 0),
            (x, size),
            80,
            1
        )

    for y in range(5, size, 100):

        cv2.line(
            image,
            (0, y),
            (size, y),
            80,
            1
        )

    # Add realistic noise

    noise = np.random.normal(
        0,
        5,
        image.shape
    )

    image = image.astype(np.float32) + noise

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    # Optical blur

    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    return image


def generate_sample(index, output_dir=OUTPUT_DIR, architecture="DRAM"):

    search = create_wafer_pattern(architecture=architecture)

    half = TARGET_SIZE // 2

    # Choose a valid target location

    x = random.randint(
        half + 10,
        IMAGE_SIZE - half - 10
    )

    y = random.randint(
        half + 10,
        IMAGE_SIZE - half - 10
    )

    # Extract target

    target = search[
        y - half:y + half,
        x - half:x + half
    ]

    # Reference is exactly 10x smaller

    reference = cv2.resize(
        target,
        (TARGET_SIZE // 10, TARGET_SIZE // 10),
        interpolation=cv2.INTER_AREA
    )

    # Simulated navigation drift

    # Sample drift first, then choose a target location that remains fully
    # inside the final (drifted) search image. This guarantees the saved
    # search image contains the target region and that the ground-truth
    # coordinates are correct after applying the drift transform.
    attempt = 0
    while True:
        attempt += 1
        drift_x = random.randint(-20, 20)
        drift_y = random.randint(-20, 20)

        # After applying drift, the true target location will be (x + drift_x, y + drift_y)
        # Choose x,y so that the final location stays within the image bounds.
        min_x = half + 10 - drift_x
        max_x = IMAGE_SIZE - half - 10 - drift_x
        min_y = half + 10 - drift_y
        max_y = IMAGE_SIZE - half - 10 - drift_y

        if min_x <= max_x and min_y <= max_y:
            x = random.randint(int(min_x), int(max_x))
            y = random.randint(int(min_y), int(max_y))
            break

        # Extremely unlikely; retry a few times before falling back to conservative bounds
        if attempt > 10:
            # conservative selection using worst-case drift margin
            max_d = 20
            x = random.randint(half + 10 + max_d, IMAGE_SIZE - half - 10 - max_d)
            y = random.randint(half + 10 + max_d, IMAGE_SIZE - half - 10 - max_d)
            drift_x = random.randint(-max_d, max_d)
            drift_y = random.randint(-max_d, max_d)
            break

    # Extract the target from the original (pre-drift) pattern to create the reference
    target = search[
        y - half:y + half,
        x - half:x + half
    ]

    reference = cv2.resize(
        target,
        (TARGET_SIZE // 10, TARGET_SIZE // 10),
        interpolation=cv2.INTER_AREA
    )

    # Apply the drift as a translation to the search image so the saved search
    # image actually contains the navigation error.
    M = np.float32([[1, 0, drift_x], [0, 1, drift_y]])
    drifted_search = cv2.warpAffine(
        search,
        M,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Save search (drifted) and reference
    search_path = os.path.join(output_dir, f"search_{index:04d}.png")
    cv2.imwrite(search_path, drifted_search)

    reference_path = os.path.join(output_dir, f"reference_{index:04d}.png")
    cv2.imwrite(reference_path, reference)

    # Save metadata. ground_truth is the coordinate of the target in the
    # saved (drifted) search image. Also record the pre-drift target for clarity.
    metadata = {
        "sample": index,
        "ground_truth": {
            "x": x + drift_x,
            "y": y + drift_y
        },
        "pre_drift_target": {
            "x": x,
            "y": y
        },
        "drift": {
            "x": drift_x,
            "y": drift_y
        },
        "reference_scale": 10,
        "architecture": architecture.upper()
    }

    metadata_path = os.path.join(output_dir, f"metadata_{index:04d}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[{index:03d}] Target=({x},{y}) Drift=({drift_x},{drift_y}) GT=({x + drift_x},{y + drift_y})")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic DRAM-style or FinFET-style benchmark "
            "dataset (reference/search image pairs + ground-truth "
            "metadata) for APEX Drift-Sense navigation-error recovery "
            "benchmarking. This is NOT a physically accurate "
            "semiconductor process simulator - see references.md."
        )
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="DRAM",
        help="Synthetic architecture style: DRAM or FinFET (case-insensitive).",
    )

    parser.add_argument(
        "--num-pairs",
        type=int,
        default=NUM_SAMPLES,
        help="Number of reference/search pairs to generate (default: 50).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to write generated samples to (default: samples).",
    )

    args = parser.parse_args()

    architecture = args.architecture.strip().upper()
    if architecture not in ARCHITECTURES:
        parser.error(
            f"--architecture must be one of {ARCHITECTURES} "
            f"(case-insensitive), got: {args.architecture!r}"
        )
    args.architecture = architecture

    if args.num_pairs <= 0:
        parser.error("--num-pairs must be a positive integer")

    return args


def main():
    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    print(
        f"Generating {args.num_pairs} {args.architecture}-style "
        f"Drift-Sense sample pair(s) into '{args.output_dir}'..."
    )

    for i in range(args.num_pairs):

        generate_sample(i, output_dir=args.output_dir, architecture=args.architecture)

    print(
        "\nDataset generation completed."
    )


if __name__ == "__main__":

    main()