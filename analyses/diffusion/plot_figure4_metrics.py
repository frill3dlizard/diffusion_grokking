import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(metrics_path):
    records = []
    with open(metrics_path, encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {metrics_path}:{line_number}: {error}") from error
            if "global_step" not in record or "type" not in record:
                raise ValueError(f"Missing type/global_step in {metrics_path}:{line_number}")
            records.append(record)
    return records


def plot_metrics(records, output_path, title):
    train_records = [record for record in records if record["type"] == "train"]
    eval_records = [record for record in records if record["type"] == "eval"]
    if not train_records and not eval_records:
        raise ValueError("No train or eval records were found.")

    fig, (accuracy_axis, loss_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if train_records:
        train_steps = [record["global_step"] for record in train_records]
        train_losses = [record["train_loss"] for record in train_records]
        loss_axis.plot(train_steps, train_losses, alpha=0.25, color="tab:blue", label="train loss")
        if len(train_losses) > 1:
            window = min(100, len(train_losses))
            smoothed = [
                sum(train_losses[max(0, index - window + 1):index + 1]) /
                len(train_losses[max(0, index - window + 1):index + 1])
                for index in range(len(train_losses))
            ]
            loss_axis.plot(train_steps, smoothed, color="tab:blue", label="train loss (moving average)")

    if eval_records:
        eval_steps = [record["global_step"] for record in eval_records]
        train_accuracy = [record["train_accuracy"] for record in eval_records]
        valid_accuracy = [record["valid_accuracy"] for record in eval_records]
        accuracy_axis.plot(eval_steps, train_accuracy, marker="o", label="train accuracy")
        accuracy_axis.plot(eval_steps, valid_accuracy, marker="o", label="validation accuracy")

    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0, 1.05)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend()

    loss_axis.set_xlabel("Optimizer step")
    loss_axis.set_ylabel("Diffusion loss")
    loss_axis.grid(alpha=0.25)
    if train_records:
        loss_axis.legend()
    else:
        loss_axis.text(
            0.5,
            0.5,
            "No persisted train-loss records",
            transform=loss_axis.transAxes,
            ha="center",
            va="center",
        )

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot locally persisted Figure 4 diffusion training metrics."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Trainer weight directory containing metrics.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="PNG path; defaults to <run-dir>/figure4_metrics.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        checkpoint_hint = os.path.join(args.run_dir, "checkpoint_latest.pth")
        raise FileNotFoundError(
            f"No metrics file found at {metrics_path}. "
            "This run predates local metric logging; rerun the recipe to create "
            f"metrics.jsonl. Checkpoint present: {os.path.isfile(checkpoint_hint)}."
        )

    records = load_metrics(metrics_path)
    output_path = args.output or os.path.join(args.run_dir, "figure4_metrics.png")
    plot_metrics(records, output_path, f"Figure 4 metrics: {os.path.basename(args.run_dir)}")
    print(f"Loaded {len(records)} records from {metrics_path}")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
