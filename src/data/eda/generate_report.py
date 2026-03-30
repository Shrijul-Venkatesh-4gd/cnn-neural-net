from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.data.data_loader import OpticalDigitsDataset


def _code_block(text: str) -> str:
    return f"```text\n{text.rstrip()}\n```"


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip()}\n"


def _bullet_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _reshape_feature_vector(values: pd.Series) -> pd.DataFrame:
    grid = values.to_numpy().reshape(8, 8)
    return pd.DataFrame(
        grid,
        index=[f"row_{idx}" for idx in range(8)],
        columns=[f"col_{idx}" for idx in range(8)],
    )


def build_eda_report() -> str:
    dataset = OpticalDigitsDataset.load()
    features = dataset.get_features()
    target = dataset.get_target(encoded=True)
    df = dataset.to_frame(encoded_target=True)

    target_distribution = pd.DataFrame(
        {
            "count": target.value_counts().sort_index(),
            "percentage": (target.value_counts(normalize=True).sort_index() * 100).round(
                2
            ),
        }
    )

    feature_profile = pd.DataFrame(
        {
            "min": features.min(),
            "max": features.max(),
            "mean": features.mean(),
            "std": features.std(),
        }
    ).round(2)
    highest_mean_pixels = feature_profile.sort_values("mean", ascending=False).head(10)
    constant_pixels = feature_profile.loc[feature_profile["std"].eq(0.0)]

    sample_intensity = df.drop(columns=[dataset.target_name]).sum(axis=1)
    intensity_by_digit = (
        pd.DataFrame({"class": target, "total_ink": sample_intensity})
        .groupby("class")
        .agg(
            count=("total_ink", "size"),
            avg_total_ink=("total_ink", "mean"),
            std_total_ink=("total_ink", "std"),
        )
        .round(2)
    )

    mean_image = _reshape_feature_vector(features.mean().round(2))
    digit_means = []
    for digit in range(10):
        digit_mean_grid = _reshape_feature_vector(
            features.loc[target.eq(digit)].mean().round(2)
        )
        digit_means.append(f"### Digit {digit}")
        digit_means.append(_code_block(digit_mean_grid.to_string()))

    sections = [
        "# Optical Digits EDA Report",
        _section(
            "Overview",
            _bullet_lines(
                [
                    f"Rows: {features.shape[0]}",
                    f"Features: {features.shape[1]}",
                    f"Target: {dataset.target_name}",
                    "Task type: 10-class handwritten digit classification",
                    "Image structure: 8x8 grid flattened into 64 integer features",
                    "Pixel value range: 0 to 16 on-pixel counts per 4x4 block",
                    "Source: UCI Optical Recognition of Handwritten Digits dataset",
                ]
            ),
        ),
        _section(
            "How To Read This Dataset",
            _bullet_lines(
                [
                    "Each row is one handwritten digit sample represented as 64 block-intensity features.",
                    "The 64 columns can be reshaped into an 8x8 image for CNN input or visual inspection.",
                    "All features are already numeric, bounded, and free of missing values.",
                    "The classes are close to balanced, so only light class-weight correction is typically needed.",
                    "Feature scaling is straightforward: divide pixel counts by 16 to normalize inputs to [0, 1].",
                ]
            ),
        ),
        _section(
            "Target Distribution",
            _code_block(target_distribution.to_string()),
        ),
        _section(
            "Data Quality",
            "\n".join(
                [
                    _bullet_lines(
                        [
                            f"Missing feature values: {int(features.isna().sum().sum())}",
                            f"Missing target values: {int(target.isna().sum())}",
                            f"Duplicate feature rows: {int(features.duplicated().sum())}",
                            f"Duplicate full rows: {int(df.duplicated().sum())}",
                            f"Constant pixels: {len(constant_pixels)} ({', '.join(constant_pixels.index.tolist())})",
                        ]
                    ),
                    "",
                    _code_block(constant_pixels.to_string() or "No constant pixel features found."),
                ]
            ),
        ),
        _section(
            "Pixel Summary Highlights",
            "\n".join(
                [
                    _code_block(highest_mean_pixels.to_string()),
                    "",
                    _bullet_lines(
                        [
                            "Higher-mean pixels trace the central stroke regions where handwritten digits overlap most often.",
                            "Border pixels are sparser, which is expected after the original 32x32 images were block-compressed to 8x8.",
                            "A couple of positions are constant zeroes across the dataset, so they can be dropped without losing signal.",
                        ]
                    ),
                ]
            ),
        ),
        _section(
            "Average Image Across All Digits",
            _code_block(mean_image.to_string()),
        ),
        _section(
            "Digit-Wise Intensity Summary",
            _code_block(intensity_by_digit.to_string()),
        ),
        _section(
            "Average Pixel Grid By Digit",
            "\n".join(digit_means),
        ),
        _section(
            "Recommended Report Extensions",
            _bullet_lines(
                [
                    "Add rendered heatmaps for the average 8x8 image overall and per digit class.",
                    "Compare confusion-prone pairs like 3 vs 5 or 8 vs 9 after a first baseline model run.",
                    "Track validation performance on the predefined digit classes to confirm stratified splits stay balanced.",
                    "Show example reconstructions from low-ink and high-ink samples to catch preprocessing mistakes early.",
                    "Document the final tensor layout clearly, especially whether the CNN expects channel-last or channel-first input.",
                ]
            ),
        ),
    ]

    return "\n\n".join(section.rstrip() for section in sections).strip() + "\n"


def write_eda_report(output_path: str = "docs/data/optical_digits_eda_report.md") -> Path:
    report = build_eda_report()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return destination


def main() -> None:
    parser = ArgumentParser(
        description="Generate a markdown EDA report for the optical digits dataset."
    )
    parser.add_argument(
        "--output",
        default="docs/data/optical_digits_eda_report.md",
        help="Where to write the markdown report.",
    )
    args = parser.parse_args()

    output = write_eda_report(args.output)
    print(f"Wrote EDA report to {output}")


if __name__ == "__main__":
    main()
