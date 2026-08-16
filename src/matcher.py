import cv2
import numpy as np


SCALE = 10


def preprocess(image):
    """Light preprocessing while preserving structural information."""

    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def find_candidates(search_image, reference_image):
    """
    Downscale the search image by 10x so that it is
    comparable with the 10x-downscaled reference.
    """

    # Downscale search image by the same factor
    small_search = cv2.resize(
        search_image,
        None,
        fx=1 / SCALE,
        fy=1 / SCALE,
        interpolation=cv2.INTER_AREA
    )

    small_search = preprocess(small_search)
    reference = preprocess(reference_image)

    # Template matching at the SAME SCALE
    result = cv2.matchTemplate(
        small_search,
        reference,
        cv2.TM_CCOEFF_NORMED
    )

    # Get dimensions
    ref_h, ref_w = reference.shape

    # Find strong candidate locations
    threshold = 0.60

    locations = np.where(result >= threshold)

    candidates = []

    for y, x in zip(locations[0], locations[1]):

        center_x = (x + ref_w / 2) * SCALE
        center_y = (y + ref_h / 2) * SCALE

        score = float(result[y, x])

        candidates.append({
            "x": center_x,
            "y": center_y,
            "score": score
        })

    return candidates


def choose_centre_candidate(search_image, candidates):

    if not candidates:
        return None

    height, width = search_image.shape

    image_center_x = width / 2
    image_center_y = height / 2

    # Required Drift-Sense rule:
    # choose candidate closest to image centre

    best = min(
        candidates,
        key=lambda candidate:
        (candidate["x"] - image_center_x) ** 2
        +
        (candidate["y"] - image_center_y) ** 2
    )

    return best


def solve(search_path, reference_path):

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    if search is None:
        raise FileNotFoundError(search_path)

    if reference is None:
        raise FileNotFoundError(reference_path)

    candidates = find_candidates(
        search,
        reference
    )

    result = choose_centre_candidate(
        search,
        candidates
    )

    return result, candidates


if __name__ == "__main__":

    result, candidates = solve(
        "samples/search_0000.png",
        "samples/reference_0000.png"
    )

    print("APEX DRIFT-SENSE")
    print("================")
    print(
        f"Candidates found : {len(candidates)}"
    )

    if result:

        print(
            f"Predicted X      : {result['x']:.1f}"
        )

        print(
            f"Predicted Y      : {result['y']:.1f}"
        )

        print(
            f"Match Score      : {result['score']:.4f}"
        )

    else:

        print("No candidates found.")