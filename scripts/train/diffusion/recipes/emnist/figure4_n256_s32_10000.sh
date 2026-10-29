#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TRAIN_DIT_SCRIPT_PATH="${ROOT_DIR}/scripts/train/diffusion/train_dit.sh"
MAX_STEPS=10000

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train/diffusion/recipes/emnist/figure4_n256_s32_10000.sh [--validate] [--dry_run]

--validate  Print the fixed Figure 4 configuration and exit without importing
            Python, downloading data, or starting training.
--dry_run   Run the Python trainer without initializing Weights & Biases.
EOF
}

DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --validate)
            echo "Figure 4 replication configuration:"
            echo "  dataset_name=emnist"
            echo "  p=23, train_fraction=0.9, num_images_per_class=256"
            echo "  image_size=32, patch_size=32, model_size=XL, depth=1"
            echo "  prediction_target=x, noise_schedule=linear"
            echo "  ffn_type=gelu, use_ffn_sandwich=true"
            echo "  lr=3e-4, lr_schedule_type=ws, weight_decay=0.01"
            echo "  batch_size=4096, sample_freq=1000, max_steps=${MAX_STEPS}"
            echo "  timestep_sampling_type=normal, confidence_threshold=0.8"
            echo "  safeguards: max_steps is fixed to ${MAX_STEPS}"
            exit 0
            ;;
        --dry_run)
            DRY_RUN=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${TRAIN_DIT_SCRIPT_PATH}" ]]; then
    chmod +x "${TRAIN_DIT_SCRIPT_PATH}"
fi

cd "${ROOT_DIR}"
args=(
    --dataset_name "emnist"
    --channel_size 1
    --image_size 32
    --p 23
    --train_fraction 0.9
    --num_images_per_class 256
    --source_data_confidence_threshold 0.8
    --patch_size 32
    --model_size "XL"
    --prediction_target "x"
    --noise_schedule "linear"
    --depth 1
    --ffn_type "gelu"
    --lr 3e-4
    --lr_schedule_type "ws"
    --weight_decay 0.01
    --epochs "${MAX_STEPS}"
    --max_steps "${MAX_STEPS}"
    --batch_size 4096
    --timestep_sampling_type "normal"
    --sample_freq 1000
    --use_ffn_sandwich
    --use_layer_norm
    --use_rope_2d
    --use_ffn_residual
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    args+=(--dry_run)
fi

echo "Starting capped Figure 4 workflow with ${MAX_STEPS} optimizer steps maximum."
"${TRAIN_DIT_SCRIPT_PATH}" "${args[@]}"
