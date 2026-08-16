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

The original matcher was reproduced at:

- Mean error: **344.47 px**
- Median error: **355.53 px**
- Within 5 px: **0%**
- Within 10 px: **2%**

The replacement matcher gives:

- Mean error: **305.41 px**
- Median error: **309.67 px**
- Within 5 px: **24%**
- Within 10 px: **26%**

The benchmark uses the repository's existing, unmodified `evaluation.py`.

## Run

```powershell
pip install -r requirements.txt
python evaluation.py
```

Do not replace the original `evaluation.py` when submitting unless the
hackathon specifically asks for a custom evaluator.
