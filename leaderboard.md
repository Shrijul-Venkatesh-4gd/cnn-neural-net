# Experiment Leaderboard

Auto-generated from `artifacts/experiment_runs.jsonl`.
Sorted by validation accuracy first so the test set stays reference-only during tuning.

| Rank | Run ID | Name | Model | Val Acc | Val Loss | Test Acc | Macro F1 | Best Epoch | Epochs | Batch | LR | WD | Dropout | CW | Params | Device | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20260330T101212975405Z | baseline-30ep | conv16-32-pool-64-gap-dense64 | 0.9448 | 0.1992 | 0.9502 | 0.9498 | 17 | 23 | 64 | 0.001 | 0.0001 | 0.2500 | yes | 28106 | cuda | full baseline training |
| 2 | 20260330T101047734216Z | quick-check | conv16-32-pool-64-gap-dense64 | 0.2384 | 2.2487 | 0.2242 | 0.1376 | 1 | 1 | 64 | 0.001 | 0.0001 | 0.2500 | yes | 28106 | cuda | single epoch preview |
| 3 | 20260330T100835989857Z | baseline-smoke | conv16-32-pool-64-gap-dense64 | 0.2260 | 2.2877 | 0.2375 | 0.1149 | 1 | 1 | 128 | 0.001 | 0.0001 | 0.2500 | yes | 28106 | cuda | tracking integration smoke test |

## Current Model

- Model class: `OpticalDigitsCNN`
- Signature: `conv16-32-pool-64-gap-dense64`
- Input tensor: `1x8x8` grayscale image
- Output classes: `10` digits
