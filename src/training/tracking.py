from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.training.pipeline import TrainingResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUN_HISTORY_PATH = ARTIFACTS_DIR / "experiment_runs.jsonl"
LEADERBOARD_PATH = PROJECT_ROOT / "leaderboard.md"


def _parameter_count(result: TrainingResult) -> int:
    return sum(parameter.numel() for parameter in result.model.parameters())


def _selected_epoch_metrics(result: TrainingResult) -> dict[str, float]:
    selected = result.history[result.best_epoch - 1]
    return {
        "train_loss": selected.train_loss,
        "train_accuracy": selected.train_accuracy,
        "val_loss": selected.val_loss,
        "val_accuracy": selected.val_accuracy,
    }


def _to_float_dict(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float):
            normalized[key] = round(value, 6)
        else:
            normalized[key] = value
    return normalized


def build_run_record(
    result: TrainingResult,
    *,
    run_name: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    selected_metrics = _selected_epoch_metrics(result)
    macro_avg = result.test_metrics.classification_report.get("macro avg", {})

    record = {
        "run_id": timestamp.strftime("%Y%m%dT%H%M%S%fZ"),
        "timestamp_utc": timestamp.isoformat(),
        "run_name": run_name or "",
        "notes": notes or "",
        "model_name": result.model.__class__.__name__,
        "model_signature": "conv16-32-pool-64-gap-dense64",
        "parameter_count": _parameter_count(result),
        "device": result.device,
        "epochs_requested": result.config.epochs,
        "epochs_completed": len(result.history),
        "best_epoch": result.best_epoch,
        "batch_size": result.config.batch_size,
        "learning_rate": result.config.learning_rate,
        "weight_decay": result.config.weight_decay,
        "dropout": result.config.dropout,
        "use_class_weights": result.config.use_class_weights,
        "random_state": result.config.random_state,
        "test_size": result.config.test_size,
        "val_size": result.config.val_size,
        "selected_train_loss": selected_metrics["train_loss"],
        "selected_train_accuracy": selected_metrics["train_accuracy"],
        "selected_val_loss": selected_metrics["val_loss"],
        "selected_val_accuracy": selected_metrics["val_accuracy"],
        "test_loss": result.test_metrics.loss,
        "test_accuracy": result.test_metrics.accuracy,
        "test_macro_f1": float(macro_avg.get("f1-score", 0.0)),
    }
    return _to_float_dict(record)


def _load_run_history() -> list[dict[str, Any]]:
    if not RUN_HISTORY_PATH.exists():
        return []
    with RUN_HISTORY_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_run_record(record: dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _write_leaderboard(records: list[dict[str, Any]]) -> None:
    sorted_records = sorted(
        records,
        key=lambda record: (
            -record["selected_val_accuracy"],
            record["selected_val_loss"],
            -record["test_accuracy"],
            record["run_id"],
        ),
    )

    lines = [
        "# Experiment Leaderboard",
        "",
        "Auto-generated from `artifacts/experiment_runs.jsonl`.",
        "Sorted by validation accuracy first so the test set stays reference-only during tuning.",
        "",
        "| Rank | Run ID | Name | Model | Val Acc | Val Loss | Test Acc | Macro F1 | Best Epoch | Epochs | Batch | LR | WD | Dropout | CW | Params | Device | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for rank, record in enumerate(sorted_records, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    record["run_id"],
                    record["run_name"] or "-",
                    record["model_signature"],
                    _format_float(record["selected_val_accuracy"]),
                    _format_float(record["selected_val_loss"]),
                    _format_float(record["test_accuracy"]),
                    _format_float(record["test_macro_f1"]),
                    str(record["best_epoch"]),
                    str(record["epochs_completed"]),
                    str(record["batch_size"]),
                    f"{record['learning_rate']:.6g}",
                    f"{record['weight_decay']:.6g}",
                    _format_float(record["dropout"]),
                    _format_bool(record["use_class_weights"]),
                    str(record["parameter_count"]),
                    record["device"],
                    (record["notes"] or "-").replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Current Model",
            "",
            "- Model class: `OpticalDigitsCNN`",
            "- Signature: `conv16-32-pool-64-gap-dense64`",
            "- Input tensor: `1x8x8` grayscale image",
            "- Output classes: `10` digits",
        ]
    )

    LEADERBOARD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_experiment_result(
    result: TrainingResult,
    *,
    run_name: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    record = build_run_record(result, run_name=run_name, notes=notes)
    _append_run_record(record)
    records = _load_run_history()
    _write_leaderboard(records)
    record["leaderboard_path"] = str(LEADERBOARD_PATH)
    record["run_history_path"] = str(RUN_HISTORY_PATH)
    record["config"] = asdict(result.config)
    return record
