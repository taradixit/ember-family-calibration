# Reproducibility protocol

## Clean environment

Create a fresh virtual environment, install the reviewed dependency lock, and
record the Python version, operating system, package versions, upstream source
revisions, and every command. `requirements.txt` names needed packages but is
not a lock file; exact package-version compatibility remains unresolved until a
clean-environment check. The `thrember` requirement is a VCS reference pinned
to the verified official EMBER2024 Git revision
`0ef753e81d98bf209f71b03cd331dfc190b5b54d`.

## Storage and expected inputs

Plan for more than the roughly 4 GB compressed download cited by the historical
script. The exact safe free-space requirement is unresolved because extracted
JSONL, vectorized features, temporary files, and filesystem overhead depend on
the pinned upstream release and feature representation. Measure these before
the full run and document the chosen capacity.

Expected external inputs are:

- `Win32_test.zip`, assigned to Win32, with 360,000 aggregate JSONL records;
- `Win64_test.zip`, assigned to Win64, with 120,000 aggregate JSONL records;
- `Dot_Net_test.zip`, assigned to Dot_Net, with 60,000 aggregate JSONL records;
- the released `EMBER2024_PE.model` from
  `joyce8/EMBER2024-benchmark-models`.

Each ZIP may contain multiple JSONL members. No stage assumes one JSONL per
file type.

Verified upstream revisions are:

- official EMBER2024/`thrember` Git:
  `0ef753e81d98bf209f71b03cd331dfc190b5b54d`;
- Hugging Face dataset `joyce8/EMBER2024`:
  `3d23efef7c0f0b702c5024400cfff4c3744a3832`;
- Hugging Face benchmark models `joyce8/EMBER2024-benchmark-models`:
  `e5b945dd90e1a1a1ec0cc07b3a17b52e9ba2d0c2`.

The historical model checksum is
`4252027863492ac138785c8c18576f43dad77d00faddc14e8c0072e8db419f99`.
The downloader must check a future pinned download against it. This foundation
does not claim that the current upstream model matches because no download was
performed.
Historical checksums for `test_metadata.parquet` and `pred_probs.npy` are
`83c87a558ab180d0aaf8a95fb59c4f28f149ebe9568e5f7c40ee83af07c20601`
and `e6aa9c3f57d168be864ef5c4d644ff7e7442d72ec4693fe4e94770d2ce17bad5`.
Those two artifacts came from the doubled pipeline and are audit references,
not acceptable corrected inputs.

Archive and extracted-member checksums, exact extracted JSONL member names,
compatible Python package versions, and feature dimensions remain unresolved
until the future pinned download and clean-environment run.

## Validation checkpoints

1. Download only the three named archives at the pinned dataset revision.
2. Reject unsafe ZIP paths, path traversal, links, unexpected non-JSONL members,
   repeated archive names, and repeated member paths.
3. Write a machine-readable manifest containing repository and revision,
   archive name/checksum/type, and every ordered member path/checksum/size/JSONL
   flag. Reject repeated resolved paths or an empty JSONL list.
4. Consume every JSONL in manifest order and require aggregate counts of 360,000
   Win32, 120,000 Win64, 60,000 Dot_Net, and 540,000 total before vectorization.
5. Report row count, unique hash count, hash multiplicities, exact duplicate
   records, binary label counts, and normalized file-type counts.
6. Stop on missing hashes, any duplicated hash, invalid labels, or count
   mismatch. Never deduplicate automatically.
7. Preserve manifest order while producing feature, `int32` label, and metadata
   rows. Read `y_test.dat` as `int32` and require exact label alignment.
8. Record metadata/features checksums, row count, feature count, and exact
   feature byte size in the preparation manifest. Inference must consume and
   validate this manifest, compare the LightGBM model feature count, and require
   finite predictions in `[0, 1]` with exactly the prepared row count.
9. Run aggregate analysis, then malicious-only family analysis with the reviewed
   minimum count. Record excluded families and all analysis parameters.

## Output provenance

Every corrected output should have a machine-readable sidecar recording input
paths and SHA-256 checksums, row and feature counts, selected upstream revisions,
dependency versions, command-line arguments, decision threshold, bin count,
family minimum, start/end times, and code commit identifier once a commit is
explicitly approved. Corrected outputs must live outside version control unless
small, reviewed result tables are later approved for inclusion.

Calibration fixes the predicted-class rule at `p >= 0.5`, uses confidence
`max(p, 1-p)`, and divides `[0.5, 1.0]` into equal-width bins. Intervals are
left-closed and right-open, `[lower, upper)`, except that the final interval
includes confidence 1.0. Empty bins are retained, contribute zero weight to
ECE, and are omitted from MCE. A separately configurable decision threshold may
change accuracy and family false-negative rate but never calibration. ROC AUC
uses scikit-learn's tested implementation.

For malicious-only family groups every true label is 1, so a thresholded error
is a false negative and family false-negative rate equals `1 - accuracy`.
Unusable family names are removed and review output is ordered by false-negative
rate descending, ECE descending, then family name.

No corrected experiment has been run.

## 2026-08-29 clean-environment preflight

A repository-local Python 3.12.0 environment successfully installed the pinned
EMBER2024 Git revision. A narrow `signify==0.7.1` pin was required because the
pinned source imports `SignedPEFile`, an API removed from the initially resolved
signify 0.9.2. `pip check`, all 53 synthetic tests, syntax checks, script help,
and Ruff passed.

The initial check found that LightGBM 4.7.0 could not locate the macOS OpenMP
runtime `libomp.dylib`, which also blocked top-level `thrember`. During the
runtime-remediation check, Homebrew 6.0.20 reported `libomp 22.1.8` already
installed at `/opt/homebrew/opt/libomp`, so it was not reinstalled. Fresh Python
processes then imported both LightGBM and top-level `thrember` successfully and
confirmed `PEFeatureExtractor().dim == 2568`. Runtime compatibility is complete
for this macOS arm64 host.

Metadata APIs confirmed all three pinned revisions. They report archive sizes
of 2,593,425,203 bytes for Win32, 1,176,459,716 bytes for Win64, and 220,481,493
bytes for Dot_Net, totaling 3,990,366,412 bytes. The model is reported as
3,756,042 bytes. Model LFS metadata reports the historical SHA-256, but the
model was not downloaded and its contents were not independently hashed.

At preflight time the project filesystem had 141,315,076,096 bytes
(131.61 GiB) free, passing the 50-GiB safety rule. At 540,000 rows, 2,568
features, and four bytes per feature, the expected vectorized feature file is
5,546,880,000 bytes (5.17 GiB). Extracted JSONL and temporary-workspace sizes
remain unknown. These are compatibility and capacity facts, not corrected
experimental results. Full details are in `environment/preflight.json`.

`environment/requirements-lock.txt` is a reproducibility snapshot of this
specific macOS arm64 environment. It is not a universal cross-platform lock
file. Dataset and model contents remain undownloaded and independently
unhashed, and no corrected experiment has run.
