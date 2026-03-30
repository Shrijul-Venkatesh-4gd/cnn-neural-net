from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import PreparedOpticalDigitsData, load_preprocessed_optical_digits_data
from src.models import OpticalDigitsCNN


@dataclass(frozen=True)
class CNNTrainingConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    early_stopping_patience: int = 6
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    use_class_weights: bool = True


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: dict


@dataclass
class TrainingResult:
    model: OpticalDigitsCNN
    history: list[EpochMetrics]
    test_metrics: EvaluationMetrics
    config: CNNTrainingConfig
    device: str
    best_epoch: int


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_data_loaders(
    prepared: PreparedOpticalDigitsData, *, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = TensorDataset(
        torch.from_numpy(np.array(prepared.X_train_images, copy=True)).float(),
        torch.from_numpy(np.array(prepared.y_train, copy=True)).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(np.array(prepared.X_val_images, copy=True)).float(),
        torch.from_numpy(np.array(prepared.y_val, copy=True)).long(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(np.array(prepared.X_test_images, copy=True)).float(),
        torch.from_numpy(np.array(prepared.y_test, copy=True)).long(),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


def train_one_epoch(
    model: OpticalDigitsCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_examples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += _accuracy_from_logits(logits, targets)
        total_examples += batch_size

    return running_loss / total_examples, running_correct / total_examples


def evaluate_model(
    model: OpticalDigitsCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    include_reports: bool = False,
) -> EvaluationMetrics:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_examples = 0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            running_loss += float(loss.item()) * batch_size
            running_correct += _accuracy_from_logits(logits, targets)
            total_examples += batch_size

            if include_reports:
                all_predictions.append(logits.argmax(dim=1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())

    average_loss = running_loss / total_examples
    accuracy = running_correct / total_examples

    if not include_reports:
        return EvaluationMetrics(
            loss=average_loss,
            accuracy=accuracy,
            confusion_matrix=np.empty((0, 0), dtype=np.int64),
            classification_report={},
        )

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    return EvaluationMetrics(
        loss=average_loss,
        accuracy=accuracy,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=list(range(10))),
        classification_report=classification_report(
            y_true,
            y_pred,
            labels=list(range(10)),
            output_dict=True,
            zero_division=0,
        ),
    )


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_loss(
    prepared: PreparedOpticalDigitsData,
    *,
    device: torch.device,
    use_class_weights: bool,
) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()

    weights = torch.tensor(
        [prepared.class_weights[class_index] for class_index in range(10)],
        dtype=torch.float32,
        device=device,
    )
    return nn.CrossEntropyLoss(weight=weights)


def train_optical_digits_cnn(config: CNNTrainingConfig) -> TrainingResult:
    set_random_seed(config.random_state)
    prepared = load_preprocessed_optical_digits_data(
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
        channel_last=False,
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        prepared,
        batch_size=config.batch_size,
    )

    device = _resolve_device()
    model = OpticalDigitsCNN(dropout=config.dropout).to(device)
    criterion = _build_loss(
        prepared,
        device=device,
        use_class_weights=config.use_class_weights,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[EpochMetrics] = []
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_metrics.loss,
                val_accuracy=val_metrics.accuracy,
            )
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        include_reports=True,
    )

    return TrainingResult(
        model=model,
        history=history,
        test_metrics=test_metrics,
        config=config,
        device=str(device),
        best_epoch=best_epoch,
    )
