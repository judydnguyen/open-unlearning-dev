#!/bin/bash
# LatentRMU Phase 1 ablation study.
# Each variant disables one or more Phase 1 loss components to measure their
# individual contribution. All variants use the same base hyperparameters as
# the tofu_Llama-3.2-1B-Instruct_forget01_LatentRMU_v4.8_sweep run.
# Outputs go to saves/unlearn/ablation/<variant>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
else
    echo "WARNING: .env not found — gated models may fail"
fi

export TRITON_CACHE_DIR=/tank/home/judy/.triton/autotune

# Use absolute paths — Hydra changes CWD to the output dir, which would break
# relative paths when model/__init__.py calls os.path.exists() on the checkpoint.
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

model="Llama-3.2-1B-Instruct"
forget_split="${1:-forget01}"
OVERWRITE=false
for arg in "$@"; do
    case "$arg" in
        -f|--overwrite|--force) OVERWRITE=true ;;
    esac
done
case "$forget_split" in
    forget01) holdout_split="holdout01"; retain_split="retain99" ;;
    forget05) holdout_split="holdout05"; retain_split="retain95" ;;
    forget10) holdout_split="holdout10"; retain_split="retain90" ;;
    *) echo "ERROR: unknown forget_split '$forget_split' (expected forget01/05/10)"; exit 1 ;;
esac
model_path="open-unlearning/tofu_${model}_full"
retain_logs_path="${PROJECT_ROOT}/saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

# Base overrides — exact match to the active run in test_latent_rmu.sh (forget01, v4.8).
# All ablation variants inherit these; only the component weights are overridden per entry.
BASE_OVERRIDES=(
    trainer.args.per_device_train_batch_size=4
    trainer.args.gradient_accumulation_steps=1
    trainer.args.num_train_epochs=12
    trainer.args.learning_rate=1e-5
    trainer.args.logging_steps=10
    trainer.args.eval_strategy=epoch
    trainer.args.save_strategy=no
    "trainer.method_args.module_regex=model\\.layers\\.7"
    trainer.method_args.encoder_epochs=6
    trainer.method_args.steering_coeff=10
    trainer.method_args.latent_dim=256
    trainer.method_args.orth_weight=5.0
    trainer.method_args.retain_sep_weight=0.0
    trainer.method_args.anchor_weight=1.0
    trainer.method_args.forget_warmup_steps=10
    trainer.method_args.gamma=1.0
    trainer.method_args.alpha=2.0
    trainer.method_args.retain_loss_type=EMBED_DIFF
)

# Format: "suffix [component_overrides...]"
# Each entry overrides only the Phase 1 loss weights relative to BASE_OVERRIDES.
#
# Phase 1 loss in code = orth_weight * grad_conflict
#                      + retain_sep_weight * retain_sep
#                      + anchor_weight * anchor
#
# Final method (BASE_OVERRIDES) sets retain_sep_weight=0 — earlier sweep showed
# it hurts forget_quality. To turn retain_sep back on for an ablation, override
# trainer.method_args.retain_sep_weight=2.0 (or whatever value) per variant.
#
# Remove-one design: 4 variants isolating each active component (retain_sep is
# already off, so no separate "no_retain_sep" variant needed).
ablations=(
    "full"
    # Phase 2 = num_train_epochs - encoder_epochs. Drop num_train_epochs to 6 so
    # no_phase1 still gets exactly 6 Phase 2 epochs (matching the other variants).
    "no_phase1      trainer.method_args.encoder_epochs=0 trainer.args.num_train_epochs=6"
    "no_orth        trainer.method_args.orth_weight=0.0"
    "no_anchor      trainer.method_args.anchor_weight=0.0"
    "with_retain_sep trainer.method_args.retain_sep_weight=2.0"
)

for ablation in "${ablations[@]}"; do
    suffix=$(echo "$ablation" | awk '{print $1}')
    # cut -f2- returns the whole string when there's no delimiter; guard against that
    if [ "$ablation" = "$suffix" ]; then
        overrides=""
    else
        overrides=$(echo "$ablation" | cut -d' ' -f2-)
    fi

    task_name="LatentRMU_${suffix}_${forget_split}"
    save_dir="${PROJECT_ROOT}/saves/unlearn/ablation/${task_name}"

    echo "=== Ablation: ${suffix} ==="
    echo "    Overrides: ${overrides:-none}"
    echo "    Save dir:  ${save_dir}"

    if [ "$OVERWRITE" = false ] && { [ -f "${save_dir}/best/model.safetensors" ] || [ -f "${save_dir}/last/model.safetensors" ]; }; then
        echo "    Checkpoint exists — skipping training (pass -f to overwrite)"
    else
        CUDA_VISIBLE_DEVICES=1 python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=LatentRMU_ablation \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            task_name=${task_name} \
            paths.output_dir=${save_dir} \
            retain_logs_path=${retain_logs_path} \
            "${BASE_OVERRIDES[@]}" \
            ${overrides}
    fi

    # Prefer best checkpoint; fall back to last; skip eval if neither exists
    if [ -f "${save_dir}/best/model.safetensors" ]; then
        ckpt="${save_dir}/best"
    elif [ -f "${save_dir}/last/model.safetensors" ]; then
        ckpt="${save_dir}/last"
    else
        echo "    No checkpoint found — skipping eval (training may have crashed)"
        echo "    Results: ${save_dir}/evals/TOFU_SUMMARY.json"
        continue
    fi

    if [ "$OVERWRITE" = false ] && [ -f "${save_dir}/evals/TOFU_SUMMARY.json" ]; then
        echo "    Eval exists — skipping (pass -f to overwrite)"
    else
        echo "    Evaluating from ${ckpt} ..."
        CUDA_VISIBLE_DEVICES=1 python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            model=${model} \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=${ckpt} \
            paths.output_dir=${save_dir}/evals \
            retain_logs_path=${retain_logs_path} \
            eval.tofu.overwrite=true
    fi

    echo "    Results: ${save_dir}/evals/TOFU_SUMMARY.json"
done
