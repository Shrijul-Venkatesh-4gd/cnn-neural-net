# cnn-neural-net

Minimal PyTorch CNN training project for the UCI Optical Recognition of Handwritten Digits dataset.

## What This Project Does

- Loads the optical digits dataset from UCI
- Preprocesses each sample into an `8x8` grayscale image
- Trains a compact CNN in PyTorch
- Evaluates with accuracy, confusion matrix, and classification report
- Logs every successful run into a simple leaderboard for lightweight experiment tracking

## Model

The current model is a small CNN designed for `8x8` inputs:

```text
1x8x8
-> Conv(16) + ReLU
-> Conv(32) + ReLU
-> MaxPool
-> Conv(64) + ReLU
-> AdaptiveAvgPool
-> Linear(64) + ReLU + Dropout
-> Linear(10)
```

## Setup

Install dependencies with:

```bash
uv sync
```

## Train Once

To run a quick one-epoch training pass:

```bash
uv run main.py --epochs 1 --run-name "quick-check" --notes "single epoch preview"
```

This will:

- train for one epoch
- print test loss and test accuracy
- print the confusion matrix
- print the classification report
- append the run to the experiment log
- regenerate the leaderboard

## Full Baseline Run

For a more realistic baseline:

```bash
uv run main.py --epochs 30 --run-name "baseline-30ep" --notes "full baseline training"
```

## Experiment Tracking

Every successful run updates:

- `leaderboard.md`: human-readable ranking of runs
- `artifacts/experiment_runs.jsonl`: append-only machine-readable run history

The leaderboard includes:

- model signature
- parameter count
- validation accuracy and loss
- test accuracy
- macro F1
- best epoch
- batch size
- learning rate
- weight decay
- dropout
- class-weight usage
- device
- notes

## Agent Workflow

The root file `agent instructions.md` contains the lightweight experiment policy for an agent:

- change one idea at a time
- optimize on validation metrics first
- keep the model compact
- prefer simple tuning before architecture changes

## Project Layout

```text
main.py
src/data/
src/models/
src/training/
agent instructions.md
leaderboard.md
artifacts/experiment_runs.jsonl
```

## Notes

- The current data loader fetches the dataset from UCI, so internet access is required unless you add a local cache path.
- A one-epoch run is only a smoke test; it is expected to classify poorly.
