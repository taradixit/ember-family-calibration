# Project context

This research project studies calibration failures in the released EMBER2024
PE detector. The current repository is maintained by Tara Dixit.

## Preliminary analysis and correction

An earlier analysis reported accuracy 0.980322, ROC AUC 0.998188, Brier score
0.014795, 15-bin ECE 0.003085, and MCE 0.027725. Those outputs came from an
invalid doubled pipeline and were not acceptable corrected evidence.

Data validation found 1,080,000 metadata rows but only 539,940 unique SHA-256
hashes. The documented PE test split contains 540,000 records. This repository
therefore treats the earlier counts and metrics as audit inputs, not final
evidence.

## Current scope

The corrected evaluation uses the reviewed 540,000-row preparation and the
released model through an inference-to-preparation manifest chain. Independent
recalculation confirms accuracy 0.980322, ROC AUC 0.998188, Brier score
0.014795, 15-bin ECE 0.003085, and MCE 0.027725. Although these rounded values
match the preliminary aggregate values, only the corrected run supports the
tracked results. The repository does not silently change or deduplicate source
data.

## Model explanation scope

The previous SHAP pipeline used EMBER2018 malicious samples and a separately
trained Random Forest. It did not explain the released EMBER2024 LightGBM
detector. That pipeline is not used here. Corrected SHAP was intentionally not
run, and SHAP does not support the new findings. Any future explanation work
must use the released detector and aligned EMBER2024 inputs.
