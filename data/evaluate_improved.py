import os
import json
import cv2
import math
import argparse

from src.matcher import solve

SAMPLES_DIR = "samples"
NUM_SAMPLES = 50


def calculate_error(predicted, ground_truth):
    dx = predicted["x"] - ground_truth["x"]
    dy = predicted["y"] - ground_truth["y"]
    return math.sqrt(dx * dx + dy * dy)


def evaluate(samples_dir, num_samples):
    per_sample = []

    print("APEX DRIFT-SENSE EVALUATION (improved)")
    print("======================================")

    for i in range(num_samples):
        search_path = os.path.join(samples_dir, f"search_{i:04d}.png")
        reference_path = os.path.join(samples_dir, f"reference_{i:04d}.png")
        metadata_path = os.path.join(samples_dir, f"metadata_{i:04d}.json")

        if not os.path.exists(search_path) or not os.path.exists(reference_path) or not os.path.exists(metadata_path):
            print(f"[{i:02d}] MISSING FILES - skipping")
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        ground_truth = metadata.get("ground_truth")
        architecture = metadata.get("architecture", "UNKNOWN")

        result, candidates = solve(search_path, reference_path)

        if result is None:
            print(f"[{i:02d}] NO MATCH - arch={architecture}")
            per_sample.append({
                "index": i,
                "architecture": architecture,
                "returned": False,
                "error": None,
                "candidates": len(candidates),
                "ground_truth": ground_truth,
                "result": None,
            })
            continue

        error = calculate_error(result, ground_truth)

        print(
            f"[{i:02d}] arch={architecture} GT=({ground_truth['x']},{ground_truth['y']}) "
            f"Pred=({result['x']:.1f},{result['y']:.1f}) Error={error:.2f}px "
            f"Candidates={len(candidates)}"
        )

        per_sample.append({
            "index": i,
            "architecture": architecture,
            "returned": True,
            "error": error,
            "candidates": len(candidates),
            "ground_truth": ground_truth,
            "result": result,
            "confidence": result.get("confidence") if isinstance(result, dict) else None,
            "chosen_by": result.get("chosen_by") if isinstance(result, dict) else None,
        })

    def summarize(samples):
        total = len(samples)
        returned = [s for s in samples if s["returned"]]
        returned_count = len(returned)
        errors = [s["error"] for s in returned]

        summary = {"total_samples": total, "returned_count": returned_count}

        if returned_count > 0:
            mean_error = sum(errors) / len(errors)
            median_error = sorted(errors)[len(errors) // 2]
            within_5 = sum(e <= 5 for e in errors)
            within_10 = sum(e <= 10 for e in errors)

            buckets = {"0-1": 0, "1-2": 0, "2-5": 0, "5-10": 0, "10-20": 0, ">20": 0}
            for e in errors:
                if e <= 1:
                    buckets["0-1"] += 1
                elif e <= 2:
                    buckets["1-2"] += 1
                elif e <= 5:
                    buckets["2-5"] += 1
                elif e <= 10:
                    buckets["5-10"] += 1
                elif e <= 20:
                    buckets["10-20"] += 1
                else:
                    buckets[">20"] += 1

            ambiguous = sum(1 for s in returned if s.get("confidence") is not None and s.get("confidence") < 0.02)
            avg_candidates = sum(s.get("candidates",0) for s in samples) / len(samples) if samples else 0

            summary.update({
                "mean_error": mean_error,
                "median_error": median_error,
                "within_5": within_5,
                "within_10": within_10,
                "error_buckets": buckets,
                "ambiguous": ambiguous,
                "avg_candidates": avg_candidates,
            })
        else:
            summary.update({
                "mean_error": None,
                "median_error": None,
                "within_5": 0,
                "within_10": 0,
                "error_buckets": {},
            })

        return summary

    arch_groups = {}
    for s in per_sample:
        arch = s["architecture"]
        arch_groups.setdefault(arch, []).append(s)

    overall = summarize(per_sample)

    print("\n============================")
    print("OVERALL SUMMARY")
    print("----------------------------")
    print(f"Samples considered   : {overall['total_samples']}")
    print(f"Matcher returned a result : {overall['returned_count']}/{overall['total_samples']}")

    if overall['mean_error'] is not None:
        print(f"Mean pixel error     : {overall['mean_error']:.2f}px")
        print(f"Median pixel error   : {overall['median_error']:.2f}px")
        print(f"Within 5 px (of those returned)  : {overall['within_5']}/{overall['returned_count']} ({overall['within_5']/overall['returned_count']*100:.2f}%)")
        print(f"Within 10 px (of those returned) : {overall['within_10']}/{overall['returned_count']} ({overall['within_10']/overall['returned_count']*100:.2f}%)")

        print("Error distribution (counts for returned results):")
        for b, c in overall['error_buckets'].items():
            print(f"  {b:6s} : {c}")
    else:
        print("No returned results to summarize errors.")

    print("\nPer-architecture breakdown:")
    for arch, samples in arch_groups.items():
        summary = summarize(samples)
        print(f"--- {arch} ---")
        print(f"Samples: {summary['total_samples']}")
        print(f"Returned: {summary['returned_count']}")
        if summary['mean_error'] is not None:
            print(f"Mean error: {summary['mean_error']:.2f}px, Median: {summary['median_error']:.2f}px")
            print(f"Within5: {summary['within_5']}/{summary['returned_count']}  Within10: {summary['within_10']}/{summary['returned_count']}")
            print("Buckets:")
            for b, c in summary['error_buckets'].items():
                print(f"  {b:6s} : {c}")
        else:
            print("No returned results for this architecture.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the matcher on a generated samples directory")
    parser.add_argument("--samples-dir", type=str, default=SAMPLES_DIR, help="Directory containing samples (default: samples)")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of samples to evaluate (default: 50)")
    args = parser.parse_args()
    evaluate(args.samples_dir, args.num_samples)
