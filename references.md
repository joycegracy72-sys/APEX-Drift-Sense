# References

This document lists supporting references for design choices made in
`data/generate_dataset.py` (the synthetic DRAM-style / FinFET-style
benchmark generator) and in the image matching / augmentation pipeline
used elsewhere in this repository.

**Important limitation:** the synthetic generator in this repository
produces simplified 2D grayscale patterns (repeated rectangular cells,
directional line structures, interconnect grids, Gaussian noise, and
blur) intended only as visually structured, repetitive benchmark
imagery for navigation-error recovery testing. It is **not** a
physically accurate semiconductor process simulator, and the references
below are cited as general background and justification for the *style*
of structure chosen (repetitive arrays for DRAM-like layouts,
directional fin structures for FinFET-like layouts, and standard image
degradation models), not as evidence that the generated images are
physically realistic reproductions of real wafer or device geometry.

## 1. Semiconductor / architecture references

These sources motivate the two high-level visual motifs used by the
generator (`--architecture DRAM` and `--architecture FinFET`):
repetitive cell-array structures for DRAM, and directional fin/gate
structures for FinFET.

- Kang, S.-M., & Leblebici, Y. *CMOS Digital Integrated Circuits:
  Analysis and Design.* McGraw-Hill. — Standard reference describing
  DRAM cell-array layout as a periodic grid of storage cells connected
  by orthogonal word lines and bit lines, which motivates the repeated
  rectangular-cell-plus-interconnect-grid motif used for the DRAM
  architecture option.

- Colinge, J.-P. (Ed.). (2008). *FinFETs and Other Multi-Gate
  Transistors.* Springer. ISBN 978-0-387-71751-7. —
  Describes the FinFET device structure as a set of parallel,
  vertically oriented silicon "fins" with a gate structure running
  across them, which motivates the parallel directional line
  ("fin") motif with a crossbar ("gate") used for the FinFET
  architecture option.

- Bohr, M., & Young, I. (2017). "CMOS Scaling Trends and Beyond." *IEEE
  Micro*, 37(6), 20–29. https://doi.org/10.1109/MM.2017.4241347 —
  General background on repetitive, array-like transistor and
  interconnect layouts in modern semiconductor process nodes.

## 2. Computer vision / image augmentation references

These sources justify the noise, blur, and template-matching choices
used in the generator and matcher pipeline.

- Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing*
  (4th ed.). Pearson. — Standard reference for additive Gaussian noise
  models and Gaussian blur as simplified models of sensor noise and
  optical defocus, used to justify `np.random.normal` noise injection
  and `cv2.GaussianBlur` in `create_wafer_pattern()`.

- Shorten, C., & Khoshgoftaar, T. M. (2019). "A survey on Image Data
  Augmentation for Deep Learning." *Journal of Big Data*, 6, 60.
  https://doi.org/10.1186/s40537-019-0197-0 — General survey covering
  noise injection and blurring as standard, lightweight image
  perturbation/augmentation techniques for building more robust and
  varied benchmark datasets.

- OpenCV Documentation: `cv2.matchTemplate` and template matching.
  https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html —
  Official documentation for the normalized cross-correlation
  (`TM_CCOEFF_NORMED`) template matching method used in
  `src/matcher.py`.

- OpenCV Documentation: `cv2.GaussianBlur`.
  https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html —
  Official documentation for the Gaussian blur filter used in both the
  generator (simulated optical blur) and the matcher (preprocessing).

---

*All citations above are to real, publicly documented sources. No
citation in this document was fabricated. These references support the
general design rationale for the synthetic benchmark; they do not
certify the generated images as physically accurate representations of
any real semiconductor device or process.*
