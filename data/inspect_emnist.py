"""Download, preprocess, and summarize the EMNIST data used by this project."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision

from data.emnist.manage_emnist_dataset import (
    EMNIST_PATH,
    MAX_EMNIST_NUM_CLASSES,
    get_emnist_image_data_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download balanced EMNIST, apply the repository preprocessing, "
            "and write metadata and representative sample images."
        )
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=23,
        help=f"Number of relabeled classes to retain (1-{MAX_EMNIST_NUM_CLASSES}).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Output height and width after the repository resize.",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=4,
        help="Number of representative processed images to save per class.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/data/emnist_inspection"),
        help="Directory for metadata.json and sample visualizations.",
    )
    return parser.parse_args()


def selected_raw_label(new_label: int) -> int:
    """Map the repository's relabeled class back to EMNIST's raw label."""
    idx_o = 10 + (ord("O") - ord("A"))
    return new_label + 1 if new_label >= idx_o else new_label


def save_samples(dataset, class_names, output_path: Path, samples_per_class: int) -> None:
    labels = dataset.tensors[1]
    images = dataset.tensors[0]
    num_classes = len(class_names)
    columns = min(5, num_classes)
    rows = (num_classes + columns - 1) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.2, rows * 2.6),
        squeeze=False,
    )
    for class_idx, class_name in enumerate(class_names):
        row, column = divmod(class_idx, columns)
        axis = axes[row][column]
        class_indices = torch.where(labels == class_idx)[0][:samples_per_class]
        image = images[class_indices].add(1).div(2).mean(dim=0).clamp(0, 1)
        axis.imshow(
            image.permute(1, 2, 0).squeeze().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axis.set_title(f"{class_idx}: {class_name}\n{len(class_indices)} shown")
        axis.axis("off")

    for unused_axis in axes.flat[len(class_names) :]:
        unused_axis.axis("off")

    figure.suptitle("Repository-preprocessed EMNIST samples", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 1 <= args.num_classes <= MAX_EMNIST_NUM_CLASSES:
        raise ValueError(f"--num_classes must be in [1, {MAX_EMNIST_NUM_CLASSES}]")
    if args.image_size < 1:
        raise ValueError("--image_size must be positive")
    if args.samples_per_class < 1:
        raise ValueError("--samples_per_class must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(EMNIST_PATH)

    print(f"Loading balanced EMNIST train split into {data_root.resolve()}")
    raw_dataset = torchvision.datasets.EMNIST(
        root=str(data_root),
        split="balanced",
        train=True,
        download=True,
    )
    processed_dataset = get_emnist_image_data_dict(
        num_classes=args.num_classes,
        data_root_dir=str(data_root),
        img_height=args.image_size,
        img_width=args.image_size,
    )

    raw_counts = Counter(raw_dataset.targets.tolist())
    processed_labels = processed_dataset.tensors[1]
    processed_counts = Counter(processed_labels.tolist())
    class_names = [
        str(raw_dataset.classes[selected_raw_label(idx)])
        for idx in range(args.num_classes)
    ]
    selected_raw_labels = [
        selected_raw_label(idx) for idx in range(args.num_classes)
    ]

    metadata = {
        "dataset": "EMNIST balanced",
        "split": "train",
        "download_root": str(data_root),
        "raw": {
            "num_samples": len(raw_dataset),
            "image_shape": list(raw_dataset.data.shape[1:]),
            "dtype": str(raw_dataset.data.dtype),
            "num_classes": len(raw_dataset.classes),
            "class_names": [str(name) for name in raw_dataset.classes],
            "class_counts": {
                str(label): count for label, count in sorted(raw_counts.items())
            },
        },
        "repository_preprocessing": {
            "num_classes": args.num_classes,
            "class_indices": list(range(args.num_classes)),
            "raw_class_indices": selected_raw_labels,
            "class_names": class_names,
            "num_samples": len(processed_dataset),
            "tensor_shape": list(processed_dataset.tensors[0].shape),
            "tensor_dtype": str(processed_dataset.tensors[0].dtype),
            "value_range": [
                float(processed_dataset.tensors[0].min()),
                float(processed_dataset.tensors[0].max()),
            ],
            "resize": [args.image_size, args.image_size],
            "horizontal_vertical_swap": True,
            "normalization": "x / 127.5 - 1.0",
            "class_counts": {
                str(label): processed_counts[label]
                for label in range(args.num_classes)
            },
        },
        "artifacts": {
            "metadata": str(output_dir / "metadata.json"),
            "samples": str(output_dir / "samples.png"),
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    save_samples(
        processed_dataset,
        class_names,
        output_dir / "samples.png",
        args.samples_per_class,
    )

    print(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {output_dir / 'metadata.json'}")
    print(f"Saved samples to {output_dir / 'samples.png'}")


if __name__ == "__main__":
    main()
