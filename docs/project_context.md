# Project context

This research project studies calibration failures in the released EMBER2024
PE detector. The current repository is maintained by Tara Dixit.

## Preliminary analysis

An earlier analysis reported accuracy 0.980322, ROC AUC 0.998188, Brier score
0.014795, 15-bin ECE 0.003085, and MCE 0.027725. These values are preliminary.
They are not corrected results.

Data validation found 1,080,000 metadata rows but only 539,940 unique SHA-256
hashes. The documented PE test split contains 540,000 records. This repository
therefore treats the earlier counts and metrics as audit inputs, not final
evidence.

## Current scope

The repository provides explicit input manifests, duplicate checks, alignment
checks, metric implementations, synthetic tests, and provenance records. It
does not silently change or deduplicate source data.

The corrected 540,000-record experiment has not run. No corrected empirical
results have been generated.

## Model explanation scope

The previous SHAP pipeline used EMBER2018 malicious samples and a separately
trained Random Forest. It did not explain the released EMBER2024 LightGBM
detector. That pipeline is not used here. Any future explanation work must use
the released detector and aligned EMBER2024 inputs.
