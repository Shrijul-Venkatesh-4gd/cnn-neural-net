from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch CNN on the optical digits dataset."
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--disable-class-weights", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from src.training import (
            CNNTrainingConfig,
            log_experiment_result,
            train_optical_digits_cnn,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is not installed in the project environment. Run `uv sync` to install dependencies."
            ) from exc
        raise

    config = CNNTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        use_class_weights=not args.disable_class_weights,
    )

    try:
        result = train_optical_digits_cnn(config)
    except ConnectionError as exc:
        raise SystemExit(
            "Downloading the optical digits dataset failed. Enable network access or cache the dataset before training."
        ) from exc

    run_record = log_experiment_result(
        result,
        run_name=args.run_name.strip() or None,
        notes=args.notes.strip() or None,
    )

    print(
        f"Training complete on {result.device}. "
        f"Best epoch: {result.best_epoch}/{config.epochs}"
    )
    print(
        f"Run logged as {run_record['run_id']} | "
        f"Leaderboard updated at {run_record['leaderboard_path']}"
    )
    print(
        f"Test loss: {result.test_metrics.loss:.4f} | "
        f"Test accuracy: {result.test_metrics.accuracy:.4f}"
    )
    print("Confusion matrix:")
    for row in result.test_metrics.confusion_matrix.tolist():
        print(row)
    print("Classification report:")
    print(json.dumps(result.test_metrics.classification_report, indent=2))


if __name__ == "__main__":
    main()
