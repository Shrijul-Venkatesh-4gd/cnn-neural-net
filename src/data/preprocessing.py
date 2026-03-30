from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_FEATURE_COUNT = 64
IMAGE_HEIGHT = 8
IMAGE_WIDTH = 8
PIXEL_MAX_VALUE = 16.0
VALID_DIGIT_CLASSES = set(range(10))


@dataclass(frozen=True)
class PreparedOpticalDigitsData:
    X_train_flat: np.ndarray
    X_val_flat: np.ndarray
    X_test_flat: np.ndarray
    X_train_images: np.ndarray
    X_val_images: np.ndarray
    X_test_images: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_weights: dict[int, float]
    raw_train: pd.DataFrame
    raw_val: pd.DataFrame
    raw_test: pd.DataFrame
    snapshot: pd.DataFrame


def validate_optical_digits_data(features: pd.DataFrame, target: pd.Series) -> None:
    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} pixel features, got {features.shape[1]}."
        )

    if features.isna().any().any():
        raise ValueError("Optical digits features contain missing values.")

    if target.isna().any():
        raise ValueError("Optical digits target contains missing values.")

    if not all(np.issubdtype(dtype, np.number) for dtype in features.dtypes):
        raise TypeError("All optical digits features must be numeric.")


def encode_digit_target(target: pd.Series) -> pd.Series:
    encoded = pd.to_numeric(target, errors="raise")
    if not np.allclose(encoded.to_numpy(), encoded.astype(int).to_numpy()):
        raise ValueError("Target contains non-integer class labels.")

    encoded = encoded.astype(int).rename(target.name or "class")
    unexpected_classes = sorted(set(encoded.unique()) - VALID_DIGIT_CLASSES)
    if unexpected_classes:
        raise ValueError(
            f"Target contains unexpected class labels: {unexpected_classes}."
        )

    return encoded


def scale_pixel_features(features: pd.DataFrame) -> pd.DataFrame:
    scaled = features.astype(np.float32) / PIXEL_MAX_VALUE
    return scaled.clip(lower=0.0, upper=1.0)


def reshape_features_to_images(
    features: np.ndarray, *, channel_last: bool = True
) -> np.ndarray:
    images = np.asarray(features, dtype=np.float32).reshape(
        -1, IMAGE_HEIGHT, IMAGE_WIDTH
    )
    if channel_last:
        return images[..., np.newaxis]
    return images[:, np.newaxis, :, :]


def _split_validation_share(test_size: float, val_size: float) -> float:
    remaining_share = 1.0 - test_size
    if remaining_share <= 0:
        raise ValueError("test_size must be less than 1.0")
    return val_size / remaining_share


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[0]
    n_classes = classes.shape[0]
    return {
        int(class_label): float(total / (n_classes * count))
        for class_label, count in zip(classes, counts, strict=True)
    }


def build_preprocessed_snapshot(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    sample_size: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    split_frames = []
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        split_frame = pd.DataFrame(
            {
                "split": split_name,
                "target": y_split.astype(int),
                "pixel_sum": X_split.sum(axis=1).round(4),
                "pixel_mean": X_split.mean(axis=1).round(4),
                "pixel_max": X_split.max(axis=1).round(4),
            }
        )
        split_frames.append(split_frame)

    combined = pd.concat(split_frames, ignore_index=True)
    n_rows = min(sample_size, len(combined))
    return combined.sample(n=n_rows, random_state=random_state).reset_index(drop=True)


def prepare_optical_digits_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    channel_last: bool = True,
) -> PreparedOpticalDigitsData:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be > 0 and sum to less than 1.")

    validate_optical_digits_data(features, target)

    encoded_target = encode_digit_target(target)
    scaled_features = scale_pixel_features(features)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        scaled_features,
        encoded_target,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_target,
    )

    val_share = _split_validation_share(test_size, val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_share,
        random_state=random_state,
        stratify=y_train_full,
    )

    X_train_flat = X_train.to_numpy(dtype=np.float32)
    X_val_flat = X_val.to_numpy(dtype=np.float32)
    X_test_flat = X_test.to_numpy(dtype=np.float32)
    y_train_array = y_train.to_numpy(dtype=np.int64)
    y_val_array = y_val.to_numpy(dtype=np.int64)
    y_test_array = y_test.to_numpy(dtype=np.int64)

    snapshot = build_preprocessed_snapshot(
        X_train=X_train_flat,
        X_val=X_val_flat,
        X_test=X_test_flat,
        y_train=y_train_array,
        y_val=y_val_array,
        y_test=y_test_array,
        sample_size=50,
        random_state=random_state,
    )

    return PreparedOpticalDigitsData(
        X_train_flat=X_train_flat,
        X_val_flat=X_val_flat,
        X_test_flat=X_test_flat,
        X_train_images=reshape_features_to_images(
            X_train_flat, channel_last=channel_last
        ),
        X_val_images=reshape_features_to_images(X_val_flat, channel_last=channel_last),
        X_test_images=reshape_features_to_images(
            X_test_flat, channel_last=channel_last
        ),
        y_train=y_train_array,
        y_val=y_val_array,
        y_test=y_test_array,
        feature_names=scaled_features.columns.tolist(),
        class_weights=compute_class_weights(y_train_array),
        raw_train=features.loc[X_train.index].reset_index(drop=True),
        raw_val=features.loc[X_val.index].reset_index(drop=True),
        raw_test=features.loc[X_test.index].reset_index(drop=True),
        snapshot=snapshot,
    )
