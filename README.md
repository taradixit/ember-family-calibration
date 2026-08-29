# EMBER2024 family calibration audit

An independently maintained, reproducibility-first audit of aggregate and
malware-family calibration in the released EMBER2024 PE detector.

**Research question:** Can strong aggregate calibration hide high-confidence
false negatives for specific malware families in the released EMBER2024 PE
detector?

An earlier analysis reported strong aggregate performance and calibration for
the released EMBER2024 PE LightGBM detector: accuracy 0.980322, ROC AUC
0.998188, Brier score 0.014795, 15-bin ECE 0.003085, and MCE 0.027725. These are
historical reproduced values, not corrected final results. Project lineage and
current scope are documented in `docs/project_context.md`.

An audit found 1,080,000 metadata rows but only 539,940 unique SHA-256 hashes.
Nearly every record appeared twice and 60 hashes appeared four times, while the
documented PE test split has 540,000 records (360,000 Win32, 120,000 Win64, and
60,000 .NET). The audit also found that the historical SHAP analysis used
EMBER2018 malicious samples and a separately trained Random Forest, so it did
not explain the released EMBER2024 LightGBM detector.

The corrected 540,000-record experiment has **not** been rerun. This repository
currently provides only the validation, metrics, script interfaces, tests, and
provenance conventions needed for a future clean run. It does not train a new
detector, include large artifacts, or make corrected empirical claims.

## Repository layout

- `src/ember_calibration/`: validation and metric functions
- `scripts/`: explicit, command-line-driven workflow stages
- `tests/`: synthetic tests that require no EMBER data
- `docs/`: project history and reproducibility protocol
- `results/`: documentation only until a corrected run is completed

## Local setup

The dependency set was verified in a clean macOS arm64 Python 3.12 environment.
LightGBM 4.7.0 and top-level `thrember` 0.1.0 import successfully with the
Homebrew OpenMP runtime, `PEFeatureExtractor().dim == 2568`, `pip check` passes,
and all 53 synthetic tests pass. `environment/requirements-lock.txt` captures
that host-specific environment; it is not a universal cross-platform lock file.

The official EMBER2024/`thrember` source remains pinned to its verified Git
commit in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest
```

No corrected empirical experiment has run. The verified environment establishes
runtime readiness only; it does not validate undownloaded dataset or model
contents and does not produce corrected results.

## Intended workflow (not run in this revision)

Each script has `--help`. Paths are explicit and repository-relative paths are
recommended.

```bash
python scripts/download_data.py --data-dir data/raw --model-dir models --execute
python scripts/prepare_test_data.py \
  --download-manifest data/raw/download_manifest.json \
  --output-dir data/processed \
  --execute-vectorization
python scripts/run_inference.py \
  --model models/EMBER2024_PE.model \
  --preparation-manifest data/processed/preparation_manifest.json \
  --output results/pred_probs.npy \
  --execute
python scripts/analyze_results.py \
  --metadata data/processed/test_metadata.parquet \
  --predictions results/pred_probs.npy \
  --output-dir results/corrected
```

The downloader requests the three verified archive names and records every
safely extracted JSONL member in order. Preparation consumes that manifest and
derives the feature count; inference consumes the resulting preparation
manifest rather than a manually supplied feature count. Downloading,
vectorization, and inference remain explicit opt-in actions.

Verified upstream pins:

- EMBER2024/`thrember` Git revision:
  `0ef753e81d98bf209f71b03cd331dfc190b5b54d`
- Hugging Face EMBER2024 dataset revision:
  `3d23efef7c0f0b702c5024400cfff4c3744a3832`
- Hugging Face benchmark-model revision:
  `e5b945dd90e1a1a1ec0cc07b3a17b52e9ba2d0c2`

## Validation policy

The pipeline never silently deduplicates. It prints a validation report with
row count, unique hash count, hash multiplicities, exact duplicate count, label
counts, and normalized file-type counts. Duplicate hashes, unexpected counts,
invalid labels or predictions, and metadata/prediction misalignment stop the
pipeline and require an explicit investigation and user decision.

Calibration uses the standard binary predicted label at `p >= 0.5`, confidence
`max(p, 1-p)`, and equal-width bins spanning `[0.5, 1.0]`. Bins use
`[lower, upper)` intervals except the last, which includes 1.0. The configurable
decision threshold affects accuracy and family false-negative rate only; it
does not redefine calibration. Empty bins appear with count zero and null
accuracy/confidence/gap, contribute zero weight to ECE, and are excluded from
MCE. ROC AUC uses `sklearn.metrics.roc_auc_score`, avoiding pairwise Python
comparisons.

Family output excludes null, blank, `unknown`, `none`, and `nan` family names
case-insensitively. It is sorted by false-negative rate descending, then ECE
descending, then family name for deterministic review. In malicious-only family
groups, accuracy and false-negative rate contain the same thresholded
information.

See [docs/reproducibility.md](docs/reproducibility.md) for the complete protocol.
