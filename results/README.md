# Results directory

This directory contains the reviewed small outputs from the corrected
540,000-row evaluation:

- aggregate metrics and 15 reliability bins;
- malicious-only family metrics at a minimum count of 100;
- calibration-bin and family-minimum sensitivity tables;
- the reliability and family-failure figures; and
- a checksummed analysis manifest tied to the implementation and inference.

The aggregate results are accuracy 0.980322, ROC AUC 0.998188, Brier score
0.014795, 15-bin ECE 0.003085, and MCE 0.027725. Family results show that these
aggregate values hide large subgroup failures. See `docs/results.md` for the
interpretation and limitations.

Large predictions, metadata, features, models, selected files, and raw data
remain ignored and are not redistributed here.
