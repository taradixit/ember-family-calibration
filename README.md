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

Data validation found why the released archives contain 1,080,000 rows. Each of
the 36 weekly JSONL members contains two ordered halves. Corresponding rows have
the same SHA-256, label, family, file type, week ID, and all 12 inputs used by
the PE detector. Only `caps`, `mbc`, and `ttps` differ. Those fields are not PE
detector inputs.

The reviewed selection takes the first documented half of each weekly member.
This is a deterministic structural choice, not a claim that the first half is
higher quality. It produces the documented 540,000 rows: 360,000 Win32, 120,000
Win64, and 60,000 Dot_Net. The selection keeps 60 hashes that repeat across
weeks without label, family, file-type, or detector-input conflicts. A separate
unique-hash sensitivity analysis will be reported after inference.

The audit also found that the historical SHAP analysis used EMBER2018 malicious
samples and a separately trained Random Forest, so it did not explain the
released EMBER2024 LightGBM detector.

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
Homebrew OpenMP runtime, `PEFeatureExtractor().dim == 2568`, and `pip check`
passes. `environment/requirements-lock.txt` captures that host-specific
environment; it is not a universal cross-platform lock file.

The official EMBER2024/`thrember` source remains pinned to its verified Git
commit in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest
```

No corrected empirical experiment has run. Environment and artifact checks do
not produce corrected results.

## Controlled workflow

Each script has `--help`. Paths are explicit and repository-relative paths are
recommended.

```bash
python scripts/download_data.py --data-dir data/raw --model-dir models
python scripts/download_data.py \
  --data-dir data/raw \
  --model-dir models \
  --execute \
  --download-only
python scripts/download_data.py --data-dir data/raw --model-dir models --execute
python scripts/select_test_records.py \
  --download-manifest data/raw/download_manifest.json \
  --output-dir data/selected
python scripts/select_test_records.py \
  --download-manifest data/raw/download_manifest.json \
  --output-dir data/selected \
  --execute
python scripts/prepare_test_data.py \
  --selection-manifest data/selected/selection_manifest.json \
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

The first command is a dry run. It prints the pinned repositories, revisions,
filenames, sizes, checksums, and repository-relative destinations without making
network requests. `--download-only` downloads and verifies all four external
files, writes `external_artifact_manifest.json`, and does not open a ZIP. Full
download mode verifies all three archives and the model before it opens any
archive. It then records every safely extracted JSONL member in order.

The selector is also a dry run unless `--execute` is present. On execution it
rechecks every source file and paired row before writing the documented first
half through temporary files. Preparation accepts only a complete selection
manifest, rechecks the selected files and residual-repeat profile, and derives
the feature count. Inference consumes the preparation manifest rather than a
manually supplied feature count. Downloading, selection, vectorization, and
inference remain explicit opt-in actions.

Verified upstream pins:

- EMBER2024/`thrember` Git revision:
  `0ef753e81d98bf209f71b03cd331dfc190b5b54d`
- Hugging Face EMBER2024 dataset revision:
  `3d23efef7c0f0b702c5024400cfff4c3744a3832`
- Hugging Face benchmark-model revision:
  `e5b945dd90e1a1a1ec0cc07b3a17b52e9ba2d0c2`

## Validation policy

The pipeline never silently deduplicates. Generic input validation still rejects
duplicate hashes. The narrow official-selection path accepts only the reviewed
profile of 539,940 unique hashes, including 60 hashes that each occur twice
across weeks. Any different multiplicity, repeated-hash-list digest, metadata
conflict, detector-input conflict, count, checksum, label, or alignment stops
the pipeline.

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
