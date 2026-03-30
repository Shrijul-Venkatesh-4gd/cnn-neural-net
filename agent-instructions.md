# Agent Instructions

## Goal
Improve the handwritten-digit CNN with small, disciplined experiments.

## Operating Style
- Keep the project minimal.
- Change one main idea per run.
- Prefer tuning hyperparameters before changing architecture.
- Optimize for validation performance first.
- Treat test metrics as reference-only, not the main selection signal.

## Experiment Loop
1. Read `leaderboard.md` and `artifacts/experiment_runs.jsonl`.
2. Pick one next hypothesis.
3. Run one controlled experiment with a clear `--run-name` and short `--notes`.
4. Compare the new run against the leaderboard.
5. Keep changes that improve validation quality or meaningfully simplify the model.
6. If a code change hurts and adds no lasting value, revert it before the next run.

## What To Tune First
- Learning rate
- Weight decay
- Dropout
- Batch size
- Early stopping patience
- Class weights on or off

Only after that:
- Small channel-count changes
- One extra convolution or one simplification

## Guardrails
- Keep the model in PyTorch.
- Keep the CNN compact for `8x8` inputs.
- Do not introduce large backbones, AutoML frameworks, or heavy experiment platforms.
- Do not change the dataset or split logic unless the experiment is explicitly about data handling.
- Record every successful run through `main.py` so the leaderboard stays current.

## Run Command
```bash
.venv/bin/python main.py --run-name "<short-name>" --notes "<what changed>" --epochs 30
```

## Success Criteria
- Primary: better validation accuracy
- Secondary: lower validation loss
- Tiebreakers: better test accuracy, better macro F1, fewer parameters, simpler config

## Artifacts
- Human-readable ranking: `leaderboard.md`
- Append-only run history: `artifacts/experiment_runs.jsonl`
