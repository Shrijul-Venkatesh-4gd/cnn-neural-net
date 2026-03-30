from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.data.preprocessing import (
    PreparedOpticalDigitsData,
    encode_digit_target,
    prepare_optical_digits_data,
    reshape_features_to_images,
    scale_pixel_features,
)


OPTICAL_DIGITS_DATASET_ID = 80


@dataclass(frozen=True)
class OpticalDigitsDataset:
    metadata: dict
    variables: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series

    @classmethod
    def load(cls, dataset_id: int = OPTICAL_DIGITS_DATASET_ID) -> "OpticalDigitsDataset":
        dataset = fetch_ucirepo(id=dataset_id)
        target = dataset.data.targets.iloc[:, 0].copy()

        return cls(
            metadata=dataset.metadata,
            variables=dataset.variables.copy(),
            features=dataset.data.features.copy(),
            target=target,
        )

    @property
    def target_name(self) -> str:
        return self.target.name or "class"

    def get_variables(self) -> pd.DataFrame:
        return self.variables.copy()

    def get_features(self) -> pd.DataFrame:
        return self.features.copy()

    def get_target(self, *, encoded: bool = False) -> pd.Series:
        target = self.target.copy()
        if encoded:
            return encode_digit_target(target)
        return target.rename(self.target_name)

    @property
    def frame(self) -> pd.DataFrame:
        return self.to_frame()

    @property
    def image_shape(self) -> tuple[int, int]:
        return (8, 8)

    def to_frame(self, *, encoded_target: bool = False) -> pd.DataFrame:
        dataset = self.get_features()
        dataset[self.target_name] = self.get_target(encoded=encoded_target)
        return dataset

    def as_images(
        self,
        *,
        normalized: bool = True,
        channel_last: bool = True,
    ) -> np.ndarray:
        features = self.get_features()
        if normalized:
            features = scale_pixel_features(features)
        return reshape_features_to_images(
            features.to_numpy(dtype=np.float32),
            channel_last=channel_last,
        )

    def preprocess_for_cnn(
        self,
        *,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        channel_last: bool = True,
    ) -> PreparedOpticalDigitsData:
        return prepare_optical_digits_data(
            self.get_features(),
            self.get_target(encoded=True),
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            channel_last=channel_last,
        )


def load_optical_digits_dataset() -> OpticalDigitsDataset:
    return OpticalDigitsDataset.load()


def load_preprocessed_optical_digits_data(
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    channel_last: bool = True,
) -> PreparedOpticalDigitsData:
    return OpticalDigitsDataset.load().preprocess_for_cnn(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        channel_last=channel_last,
    )
