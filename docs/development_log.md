# Development log

## 2026-08-29 — initial reproducibility work

- Recorded that preliminary metadata contains 1,080,000 rows and 539,940 unique
  SHA-256 hashes. Nearly every hash occurs twice, and 60 hashes occur four times.
- Classified the reported aggregate metrics as preliminary values rather than
  corrected results.
- Added validation that reports input counts and stops on duplicate hashes.
- Recorded that the previous SHAP pipeline used EMBER2018 data and a separate
  Random Forest, so it did not explain the released EMBER2024 LightGBM detector.
- Added synthetic tests and explicit workflow interfaces. No data was downloaded
  and no real inference, training, SHAP, or corrected analysis ran.

## 2026-08-29 — validation and interface updates

- Set predicted-label calibration bins to equal widths over `[0.5, 1.0]` with a
  fixed prediction rule at `p >= 0.5`.
- Replaced the quadratic custom ROC AUC with scikit-learn's implementation.
- Set vectorized-label validation to the `thrember` `int32` representation.
- Replaced the one-JSONL-per-type interface with a pinned archive and ordered
  member manifest.
- Made inference consume a preparation manifest and validate checksums, sizes,
  dimensions, model feature count, and predictions.
- Recorded verified upstream revisions and added tests for these checks.

## 2026-08-29 — clean runtime baseline

- Verified the dependency set in a clean macOS arm64 Python 3.12 environment.
- Added the narrow `signify==0.7.1` compatibility pin required by the pinned
  `thrember` source.
- Verified the Homebrew OpenMP runtime and imports of LightGBM 4.7.0 and
  top-level `thrember` 0.1.0.
- Confirmed `PEFeatureExtractor().dim == 2568`, `pip check`, and all 53 synthetic
  tests.
- Used no dataset archives, benchmark model, generated arrays, or corrected
  experiment in establishing this baseline.
