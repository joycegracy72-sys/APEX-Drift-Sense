# APEX Drift-Sense

APEX Drift-Sense estimates the location of a small reference image inside a
larger search image when the reference is 10× downsampled.

## Core approach

1. Downsample the search image by the known 10× scale.
2. Apply normalized cross-correlation.
3. Remove redundant neighboring detections using period-aware NMS.
4. Keep only genuinely competitive high-score candidates.
5. Apply the Drift-Sense centre-distance rule only inside that quality-gated set.

## Verified benchmark on the supplied 50-sample dataset

Measured results (using the repository matcher as provided):

- Original `samples/` dataset (reported by the improved evaluator):
  - Mean error: 305.41 px
  - Median error: 309.67 px
  - Within 5 px: 24% (of returned results)
  - Within 10 px: 26% (of returned results)

Notes:
- These numbers are measured with the matcher implementation in `src/matcher.py` and the improved evaluation script `data/evaluate_improved.py` included in this change set.
- The matcher is NOT architecture-aware; it does not consume or use the `architecture` field in the dataset metadata. Reported DRAM vs FinFET comparisons are empirical measurements of the matcher running on datasets generated with different synthetic styles, not evidence of any architecture-specific matching logic.

## Run

```powershell
pip install -r requirements.txt
python evaluation.py
```

Do not replace the original `evaluation.py` when submitting unless the
hackathon specifically asks for a custom evaluator.

## Judge / Reproduction Instructions

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Generate a test dataset

The generator produces synthetic DRAM-style and FinFET-style
reference/search image pairs for navigation-error recovery
benchmarking. These are simplified, repetitive synthetic patterns
designed to be distinguishable and useful for testing - they are
**not** a physically accurate simulation of any real DRAM or FinFET
process (see [references.md](references.md) for supporting
background and limitations).

```bash
python data/generate_dataset.py \
    --architecture DRAM \
    --num-pairs 1 \
    --output-dir test_samples
```

```bash
python data/generate_dataset.py \
    --architecture FinFET \
    --num-pairs 1 \
    --output-dir test_samples_finfet
```

`--architecture` accepts `DRAM` or `FinFET` (case-insensitive). This
writes to a separate output directory and does **not** touch or
regenerate the original benchmark dataset in `samples/`, which is
required to reproduce the numbers reported above.

### 3. Run inference

```bash
python inference.py test_samples/reference_0000.png test_samples/search_0000.png
```

`inference.py` expects exactly two arguments, in this order:

1. **reference image** (first argument) - the small template image.
2. **search image** (second argument) - the larger image to locate it in.

It calls the existing, unmodified `src/matcher.py` internally and
prints the predicted location as a single coordinate, e.g.:

```
(512, 438)
```

Exit code is non-zero (with an error message on stderr) if the
arguments are missing/invalid or the input files can't be read.

### Scope note

To avoid any ambiguity: **synthetic dataset generation**
(`data/generate_dataset.py`), **image matching**
(`src/matcher.py`, unchanged), and **navigation-error recovery**
(the overall benchmarking task this repo evaluates) are three distinct
things. Generating a synthetic DRAM-style or FinFET-style image does
not imply the matcher was trained or tuned on real wafer data, and the
reported benchmark numbers above reflect performance on the original,
unmodified `samples/` dataset only.
