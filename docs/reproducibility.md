# Reproducibility protocol

## Clean environment

Create a fresh virtual environment, install the reviewed dependencies, and
record the Python version, operating system, package versions, upstream source
revisions, and every command. `requirements.txt` names the direct dependencies.
`environment/requirements-lock.txt` records the verified macOS arm64 environment
and is not a universal lock file. The `thrember` requirement is a VCS reference
pinned to the verified official EMBER2024 Git revision
`0ef753e81d98bf209f71b03cd331dfc190b5b54d`.

## Storage and expected inputs

Plan for more than the roughly 4 GB compressed download cited by the historical
script. The exact safe free-space requirement is unresolved because extracted
JSONL, vectorized features, temporary files, and filesystem overhead depend on
the pinned upstream release and feature representation. Measure these before
the full run and document the chosen capacity.

Expected external inputs are:

- `Win32_test.zip`, assigned to Win32;
- `Win64_test.zip`, assigned to Win64;
- `Dot_Net_test.zip`, assigned to Dot_Net;
- the released `EMBER2024_PE.model` from
  `joyce8/EMBER2024-benchmark-models`.

Each ZIP contains 12 weekly JSONL members. The released members contain
1,080,000 rows because every member has two ordered halves. The documented
selection contains 360,000 Win32, 120,000 Win64, and 60,000 Dot_Net rows.

Verified upstream revisions are:

- official EMBER2024/`thrember` Git:
  `0ef753e81d98bf209f71b03cd331dfc190b5b54d`;
- Hugging Face dataset `joyce8/EMBER2024`:
  `3d23efef7c0f0b702c5024400cfff4c3744a3832`;
- Hugging Face benchmark models `joyce8/EMBER2024-benchmark-models`:
  `e5b945dd90e1a1a1ec0cc07b3a17b52e9ba2d0c2`.

Metadata at the pinned revisions reports these expected artifacts:

- `Win32_test.zip`: 2,593,425,203 bytes and SHA-256
  `c05f6562dee3ace4195087be918eb00181e33bc31464c671fb5ba00c9dd5dfdb`;
- `Win64_test.zip`: 1,176,459,716 bytes and SHA-256
  `52a5a05c1bfa5bb021bb8b44c2e0afcf8983dfa1c6c0a9d76db393e5c682ce10`;
- `Dot_Net_test.zip`: 220,481,493 bytes and SHA-256
  `b74c4181dbd77565fce16ba47c8ab0f7c7044ae6d880d859ec2c27365dea6299`;
- `EMBER2024_PE.model`: 3,756,042 bytes and SHA-256
  `4252027863492ac138785c8c18576f43dad77d00faddc14e8c0072e8db419f99`.

The historical model checksum is
`4252027863492ac138785c8c18576f43dad77d00faddc14e8c0072e8db419f99`.
The metadata value is the same as the digest confirmed during the controlled
local download check. Large artifacts remain ignored and are not part of the
repository.
Historical checksums for `test_metadata.parquet` and `pred_probs.npy` are
`83c87a558ab180d0aaf8a95fb59c4f28f149ebe9568e5f7c40ee83af07c20601`
and `e6aa9c3f57d168be864ef5c4d644ff7e7442d72ec4693fe4e94770d2ce17bad5`.
Those two artifacts came from the doubled pipeline and are audit references,
not acceptable corrected inputs.

The controlled extraction produced 36 verified JSONL members totaling
17,655,416,292 bytes. The extraction manifest records every member name, size,
and SHA-256 value. These large files remain outside version control.

## Reviewed record selection

Each weekly member contains exactly twice its documented count. The selection
compares corresponding rows and keeps the first documented half: 30,000 Win32,
10,000 Win64, or 5,000 Dot_Net rows per member. The first half is a deterministic
structural choice. It is not treated as higher quality than the second half.

All 540,000 paired positions match on SHA-256, label, family, file type, week
ID, and the 12 PE detector inputs. Only `caps`, `mbc`, and `ttps` differ. The
pinned `thrember 0.1.0` extractor confirms that those three fields are not used
to create its 2,568-feature PE vector.

The selected data has 539,940 unique hashes. Of these, 539,880 occur once and
60 occur twice across member and week boundaries. The repeats have no label,
family, file-type, or detector-input conflicts. The SHA-256 of the sorted
repeated-hash list is
`81c20f8d9397f4f27143652988dfdc036edc3a3c948a0efe75e7817e97283767`.
The primary analysis keeps these rows. A separate unique-hash sensitivity
analysis will be reported after inference.

## Validation checkpoints

1. Download only the three named archives and model at their pinned revisions.
2. Treat every downloaded file as untrusted. Require its exact byte size and
   SHA-256 before opening any archive.
3. Stop on the first size or checksum mismatch. Do not extract a ZIP or write a
   success manifest after a failed verification.
4. Write `external_artifact_manifest.json` with repository-relative paths and
   expected and observed integrity values. `--download-only` stops here.
5. Reject unsafe ZIP paths, path traversal, links, unexpected non-JSONL members,
   repeated archive names, and repeated member paths.
6. Write the extraction manifest with every ordered member path, checksum, size,
   type, and JSONL flag. Reject repeated resolved paths or an empty JSONL list.
7. Read every source member in manifest order. Require twice its documented
   count and compare the two halves with separate streaming handles.
8. Require paired SHA-256, label, family, file type, week ID, and all 12 detector
   inputs to match. Reject any difference outside `caps`, `mbc`, and `ttps`.
9. Write the first documented half through temporary files. Rename only after
   all members, total counts, and the reviewed residual-repeat profile pass.
10. Before preparation, require a complete selection manifest. Recheck every
    selected file's size and SHA-256 and recompute its repeat profile. Generic
    validation continues to reject duplicate hashes.
11. Preserve selection-manifest order while producing feature, `int32` label,
    and metadata rows, including `week_id`. Read `y_test.dat` as `int32` and
    require exact label alignment.
12. Record metadata/features checksums, row count, feature count, and exact
   feature byte size in the preparation manifest. Inference must consume and
   validate this manifest, compare the LightGBM model feature count, and require
   finite predictions in `[0, 1]` with exactly the prepared row count.
13. Run aggregate analysis, then malicious-only family analysis with the reviewed
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
were still unknown at that stage. The later controlled extraction measured
17,655,416,292 JSONL bytes. These are compatibility and capacity facts, not
corrected experimental results. Full preflight details are in
`environment/preflight.json`.

`environment/requirements-lock.txt` is a reproducibility snapshot of this
specific macOS arm64 environment. It is not a universal cross-platform lock
file. Later controlled checks independently verified the downloaded dataset and
model contents. No corrected experiment has run.
