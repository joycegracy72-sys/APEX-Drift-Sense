import os
import cv2
import json
import random
import numpy as np


IMAGE_SIZE = 1000
TARGET_SIZE = 100
NUM_SAMPLES = 50

OUTPUT_DIR = "samples"


def create_wafer_pattern(size=IMAGE_SIZE):

    image = np.zeros((size, size), dtype=np.uint8)

    # Repetitive semiconductor-like cells
    cell_w = 40
    cell_h = 40
    spacing = 60

    for y in range(20, size - 40, spacing):

        for x in range(20, size - 40, spacing):

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


def generate_sample(index):

    search = create_wafer_pattern()

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

    drift_x = random.randint(
        -20,
        20
    )

    drift_y = random.randint(
        -20,
        20
    )

    # Save search

    search_path = os.path.join(
        OUTPUT_DIR,
        f"search_{index:04d}.png"
    )

    cv2.imwrite(
        search_path,
        search
    )

    # Save reference

    reference_path = os.path.join(
        OUTPUT_DIR,
        f"reference_{index:04d}.png"
    )

    cv2.imwrite(
        reference_path,
        reference
    )

    # Save metadata

    metadata = {

        "sample": index,

        "ground_truth": {
            "x": x,
            "y": y
        },

        "drift": {
            "x": drift_x,
            "y": drift_y
        },

        "reference_scale": 10
    }

    metadata_path = os.path.join(
        OUTPUT_DIR,
        f"metadata_{index:04d}.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"[{index:03d}] "
        f"Target=({x},{y}) "
        f"Drift=({drift_x},{drift_y})"
    )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "Generating improved Drift-Sense dataset..."
    )

    for i in range(NUM_SAMPLES):

        generate_sample(i)

    print(
        "\nDataset generation completed."
    )


if __name__ == "__main__":

    main()