#!/usr/bin/env python3
"""Audit Figure 4 training/evaluation semantics without loading a model."""

import argparse
from collections import Counter


def build_pair_split(p, train_fraction):
    pair_groups = {}
    for a in range(p):
        for b in range(a, p):
            key = (a, b)
            pair_groups[key] = [(a, b, (a + b) % p)]
            if a != b:
                pair_groups[key].append((b, a, (a + b) % p))

    keys = list(pair_groups)
    # Match Modular_Arithmetic_Dataset's fixed NumPy seed and shuffle.
    import numpy as np

    np.random.seed(42)
    np.random.shuffle(keys)
    train_size = int(train_fraction * len(keys))
    train_keys = set(keys[:train_size])
    valid_keys = set(keys[train_size:])
    train_pairs = [pair for key in train_keys for pair in pair_groups[key]]
    valid_pairs = [pair for key in valid_keys for pair in pair_groups[key]]
    return train_keys, valid_keys, train_pairs, valid_pairs


def schedule(step, total_steps, base_lr, schedule):
    warmup_end = min(total_steps * 0.05, 10000)
    pivot = total_steps * 0.5
    if schedule == "ws":
        if step < warmup_end:
            multiplier = 0.1 + 0.9 * step / warmup_end
        else:
            multiplier = 1.0
    elif schedule == "wsd":
        if step < warmup_end:
            multiplier = 0.1 + 0.9 * step / warmup_end
        elif step < pivot:
            multiplier = 1.0
        else:
            progress = (step - pivot) / (total_steps - pivot)
            multiplier = 1.0 - 0.99 * progress
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")
    return base_lr * multiplier


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=23)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--num-images-per-class", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--paper-lr", type=float, default=5e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    train_keys, valid_keys, train_pairs, valid_pairs = build_pair_split(
        args.p, args.train_fraction
    )
    train_key_overlap = train_keys & valid_keys
    ordered_eval_pairs = args.num_images_per_class // args.p
    full_eval_size = ordered_eval_pairs * args.p * args.p
    steps_per_epoch = max(1, (len(train_pairs) + args.batch_size - 1) // args.batch_size)

    print("Figure 4 audit")
    print(f"  p={args.p}, train_fraction={args.train_fraction}")
    print(
        f"  unordered groups: train={len(train_keys)}, valid={len(valid_keys)}, "
        f"overlap={len(train_key_overlap)}"
    )
    print(f"  ordered pair samples: train={len(train_pairs)}, valid={len(valid_pairs)}")
    print(
        f"  trainer quick metric: 32 train samples + 32 validation samples "
        f"(validation loader length={len(valid_pairs)})"
    )
    print(
        f"  paper-style full evaluation: {ordered_eval_pairs} repeats * "
        f"{args.p}^2 ordered pairs = {full_eval_size} samples"
    )
    print(
        f"  steps per epoch at batch_size={args.batch_size}: {steps_per_epoch}; "
        "with this dataset, optimizer step and epoch are usually equivalent"
    )
    print("  source image pool: shared by train/validation (intentional visual diversity)")
    print("  current EMNIST class mapping: balanced labels 0-9 plus A-M (23 classes)")
    print("  paper text describes uppercase A-W; this is a repository/paper fidelity gap")
    print("  architecture: XL d=2048, 64 heads, depth=1, FFN=8192, patch_size=32, 3 tokens")
    print("  paper parameter count: approximately 106.2M")
    for schedule_name, base_lr, label in (
        ("ws", args.lr, "current 3e-4 recipe"),
        ("wsd", args.paper_lr, "paper-aligned 5e-4 recipe"),
    ):
        values = [
            schedule(step, args.total_steps, base_lr, schedule_name)
            for step in (0, 5000, 10000, args.total_steps // 2, args.total_steps)
        ]
        formatted = ", ".join(f"{value:.3e}" for value in values)
        print(f"  {label} ({schedule_name}) LR at [0,5k,10k,mid,end]: {formatted}")
    print(
        "  confirmed discrepancies: current recipe uses 3e-4 + ws and t-weighted RF loss; "
        "paper/HF evidence supports 5e-4 + decay and Appendix A specifies unweighted CFM."
    )
    print("  target labels in both datasets are computed as (a+b) mod p; no label shortcut.")
    print(f"  validation label counts in full pair split: {dict(Counter(c for _, _, c in valid_pairs))}")


if __name__ == "__main__":
    main()
