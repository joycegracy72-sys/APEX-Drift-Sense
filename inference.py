"""
inference.py

Thin command-line wrapper around the verified, frozen matcher in
src/matcher.py. This file does NOT reimplement or alter the matching
algorithm in any way - it only handles argument parsing, input
validation, and formatting the final output.

Usage:
    python inference.py reference.png search.png

Arguments:
    reference.png   The small reference/template image (first argument).
    search.png       The larger search image to locate the reference
                      within (second argument).

Output:
    Prints a single line with the predicted (x, y) coordinate, e.g.:

        (512, 438)

Exit codes:
    0   Success - a coordinate was found and printed.
    1   Invalid usage (wrong number of arguments).
    2   One or more input files do not exist / could not be read.
    3   The matcher ran but found no confident match.
"""

import os
import sys

from src.matcher import solve


def fail(message, code=1):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def main():
    if len(sys.argv) != 3:
        fail(
            "expected exactly 2 arguments.\n"
            "Usage: python inference.py reference.png search.png",
            code=1,
        )

    reference_path = sys.argv[1]
    search_path = sys.argv[2]

    for path in (reference_path, search_path):
        if not os.path.isfile(path):
            fail(f"file not found: {path}", code=2)

    try:
        # NOTE: src.matcher.solve() has the signature
        #   solve(search_path, reference_path)
        # i.e. the SEARCH image first, REFERENCE image second - the
        # opposite order to this script's own CLI. We just pass the
        # arguments through in the order solve() actually expects.
        result, candidates = solve(search_path, reference_path)
    except FileNotFoundError as e:
        fail(f"could not read image file: {e}", code=2)
    except Exception as e:
        fail(f"matcher failed: {e}", code=2)

    if result is None:
        fail(
            f"no confident match found ({len(candidates)} candidate(s) considered).",
            code=3,
        )

    x = round(result["x"])
    y = round(result["y"])
    print(f"({x}, {y})")


if __name__ == "__main__":
    main()
