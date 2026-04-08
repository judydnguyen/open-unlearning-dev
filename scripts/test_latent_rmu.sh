# #!/bin/bash

# # Test script for LatentRMU: two-phase RMU unlearning with a learned per-sample
# # steering encoder.
# #
# # Phase 1 (encoder_epochs): encoder is trained on frozen LLM activations.
# #   - Orthogonality loss: steering vectors should not lie in retain PCA subspace.
# #
# # Phase 2 (remaining epochs): LLM is fine-tuned with the fixed encoder.
# #   - Forget loss: push forget activations toward encoder-generated control vectors.
# #   - Retain loss: keep retain activations close to reference model (EMBED_DIFF).
# #
# # v4.5: Improve Phase 1 encoder training with retain separation loss.
# #       In addition to gradient-conflict orthogonality (orth_weight), add a direct
# #       signal (retain_sep_weight) that penalizes encoder output r for being aligned
# #       with retain activations (from ref_model). This trains the encoder to find
# #       forget-specific directions that are orthogonal to the retain subspace, so
# #       Phase 2 steering vectors won't perturb retain content.

# set -e

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

# MODEL="Llama-3.2-1B-Instruct"
# # FORGET_SPLIT="forget01"
# # RETAIN_SPLIT="retain99"
# # HOLDOUT_SPLIT="holdout01"
# FORGET_SPLIT="forget10"
# RETAIN_SPLIT="retain90"
# HOLDOUT_SPLIT="holdout10"
# MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
# TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_LatentRMU_v4.8
# GPUS="1"

# echo "=========================================="
# echo "Running LatentRMU unlearning"
# echo "Model: $MODEL"
# echo "Task: $TASK_NAME"
# echo "=========================================="

# # Step 1: Run Unlearning
# CUDA_VISIBLE_DEVICES=$GPUS /data/judy/conda/envs/unlearning/bin/python src/train.py --config-name=unlearn.yaml \
#     experiment=unlearn/tofu/default \
#     trainer=LatentRMU \
#     task_name=${TASK_NAME} \
#     model=${MODEL} \
#     forget_split=${FORGET_SPLIT} \
#     retain_split=${RETAIN_SPLIT} \
#     model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
#     retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
#     trainer.args.per_device_train_batch_size=4 \
#     trainer.args.gradient_accumulation_steps=1 \
#     trainer.args.num_train_epochs=9 \
#     trainer.args.learning_rate=1e-5 \
#     trainer.args.logging_steps=10 \
#     trainer.args.eval_strategy=epoch \
#     trainer.args.save_strategy=no \
#     trainer.method_args.module_regex="model\.layers\.7" \
#     trainer.method_args.encoder_epochs=4 \
#     trainer.method_args.steering_coeff=10 \
#     trainer.method_args.latent_dim=256 \
#     trainer.method_args.orth_weight=2.0 \
#     trainer.method_args.retain_sep_weight=2.0 \
#     trainer.method_args.forget_warmup_steps=30 \
#     trainer.method_args.gamma=1.0 \
#     trainer.method_args.alpha=1.0 \
#     trainer.method_args.retain_loss_type=EMBED_DIFF

# echo "=========================================="
# echo "Training completed!"
# echo "Results saved to: saves/unlearn/${TASK_NAME}"
# echo "=========================================="

# # Step 2: Evaluate
# CUDA_VISIBLE_DEVICES=${GPUS%%,*} /data/judy/conda/envs/unlearning/bin/python src/eval.py \
#     --config-name=eval.yaml \
#     experiment=eval/tofu/default \
#     forget_split=${FORGET_SPLIT} \
#     holdout_split=${HOLDOUT_SPLIT} \
#     model=${MODEL} \
#     task_name=${TASK_NAME} \
#     model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME} \
#     paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
#     retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

# echo "=========================================="
# echo "Done. Results: saves/unlearn/${TASK_NAME}/evals"
# echo "=========================================="


#!/bin/bash

# Test script for LatentRMU: two-phase RMU unlearning with a learned per-sample
# steering encoder.
#
# Phase 1 (encoder_epochs): encoder is trained on frozen LLM activations.
#   - Orthogonality loss: steering vectors should not lie in retain PCA subspace.
#
# Phase 2 (remaining epochs): LLM is fine-tuned with the fixed encoder.
#   - Forget loss: push forget activations toward encoder-generated control vectors.
#   - Retain loss: keep retain activations close to reference model (EMBED_DIFF).
#
# v4.5: Improve Phase 1 encoder training with retain separation loss.
#       In addition to gradient-conflict orthogonality (orth_weight), add a direct
#       signal (retain_sep_weight) that penalizes encoder output r for being aligned
#       with retain activations (from ref_model). This trains the encoder to find
#       forget-specific directions that are orthogonal to the retain subspace, so
#       Phase 2 steering vectors won't perturb retain content.

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

MODEL="Llama-3.2-1B-Instruct"
# FORGET_SPLIT="forget01"
# RETAIN_SPLIT="retain99"
# HOLDOUT_SPLIT="holdout01"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME=tofu_${MODEL}_${FORGET_SPLIT}_LatentRMU_v4.8
GPUS="1"

echo "=========================================="
echo "Running LatentRMU unlearning"
echo "Model: $MODEL"
echo "Task: $TASK_NAME"
echo "=========================================="

# Step 1: Run Unlearning
CUDA_VISIBLE_DEVICES=$GPUS /data/judy/conda/envs/unlearning/bin/python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=LatentRMU \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.num_train_epochs=9 \
    trainer.args.learning_rate=1e-5 \
    trainer.args.logging_steps=10 \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    trainer.method_args.module_regex="model\.layers\.7" \
    trainer.method_args.encoder_epochs=4 \
    trainer.method_args.steering_coeff=10 \
    trainer.method_args.latent_dim=256 \
    trainer.method_args.orth_weight=2.0 \
    trainer.method_args.retain_sep_weight=2.0 \
    trainer.method_args.forget_warmup_steps=30 \
    trainer.method_args.gamma=1.0 \
    trainer.method_args.alpha=2.0 \
    trainer.method_args.retain_loss_type=EMBED_DIFF

echo "=========================================="
echo "Training completed!"
echo "Results saved to: saves/unlearn/${TASK_NAME}"
echo "=========================================="

# Step 2: Evaluate
CUDA_VISIBLE_DEVICES=${GPUS%%,*} /data/judy/conda/envs/unlearning/bin/python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model=${MODEL} \
    task_name=${TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME} \
    paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
    retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

echo "=========================================="
echo "Done. Results: saves/unlearn/${TASK_NAME}/evals"
echo "=========================================="
