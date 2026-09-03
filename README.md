# Aggregate and Malware-Family Calibration in EMBER2024

I evaluated the released EMBER2024 PE detector on 540,000 records. The overall
accuracy and calibration scores are strong, but some malware families have high
false-negative rates.

The study uses the released LightGBM model and the 2,568-feature `thrember`
representation. It does not retrain or recalibrate the detector. This repository
is maintained by Tara Dixit.

[Read the paper](paper/EMBER2024_Family_Calibration.pdf)

## Method

The pipeline checks the released PE archives, selects the documented weekly
counts, builds features in source order, and runs the released model in batches.
It measures accuracy, ROC AUC, Brier score, expected calibration error (ECE), and
maximum calibration error (MCE). Family results use malicious records from
families with at least 100 examples.

## Data

The evaluation has 360,000 Win32, 120,000 Win64, and 60,000 Dot_Net records.
The released archives contain duplicated structural halves. The selected half
has 539,940 unique hashes. Sixty hashes occur twice across weeks, with no label,
family, file-type, or detector-input conflicts.

## Results

| Metric | Result |
|---|---:|
| Accuracy | 98.03% |
| ROC AUC | 0.9982 |
| Brier score | 0.0148 |
| 15-bin ECE | 0.0031 |
| 15-bin MCE | 0.0277 |

The aggregate scores hide larger errors for some families.

| Family | Malicious records | False-negative rate | 15-bin ECE |
|---|---:|---:|---:|
| `malicord` | 162 | 85.19% | 0.602 |
| `lazzzy` | 179 | 62.01% | 0.452 |
| `rugmi` | 256 | 61.72% | 0.447 |

## Limits

- Family results are descriptive. They do not show why a family has more errors.
- Family labels are missing for 37,633 malicious records.
- Small-family rankings change when the minimum record count changes.
- Sixty hashes repeat across weeks, and no unique-hash sensitivity was run.
- The study does not report confidence intervals.
- A separate SHAP pipeline used EMBER2018 data and a Random Forest, so it did not
  explain this model. Model-aligned SHAP, retraining, and post-hoc calibration
  are outside this study.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
ruff check --no-cache .
```

The workflow scripts provide `--help` and require explicit execution flags for
data preparation, inference, and analysis. Large data, models, features, and
predictions are ignored by Git.

## Files

```text
paper/        paper and LaTeX source
src/          validation and metric code
scripts/      data, inference, and analysis commands
tests/        synthetic unit tests
results/      small result tables and figures
environment/  host-specific environment records
```
