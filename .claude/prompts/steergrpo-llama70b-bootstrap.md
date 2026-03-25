Use the `research-iteration`, `experiment-safety`, and `metric-triage` skills.

Goal: test `SteerGRPO` on Llama 70B in this repo, but there is no TOFU finetuned base model for that setting yet, so first create the required finetuned TOFU model before attempting unlearning.

Work in this order:

1. Inspect the existing TOFU finetune and unlearning pipeline in:
   - `src/trainer/unlearn/SteerGRPO.py`
   - `configs/experiment/finetune/tofu/default.yaml`
   - `configs/experiment/unlearn/tofu/default.yaml`
   - relevant model configs under `configs/model/`
   - existing TOFU scripts under `scripts/`

2. Determine the best-supported Llama 70B config path in this repo.
   - If no exact 70B config exists, create the minimal config needed by adapting the nearest existing Llama config.
   - Do not redesign the method or benchmark.

3. Create a minimal, reproducible pipeline for Llama 70B TOFU:
   - finetune the base model on TOFU first
   - evaluate the retain/finetuned model and save the eval logs needed for TOFU metrics
   - then prepare a SteerGRPO unlearning run that uses that finetuned checkpoint as input

4. Before launching anything expensive, self-criticize the plan:
   - identify memory or throughput risks for 70B
   - identify config gaps
   - identify whether LoRA or another already-supported low-memory option is required
   - identify what would make the comparison invalid

5. Then make only the minimal code/config/script changes required.

6. Add or adapt scripts so the runs are traceable with fresh task names:
   - finetune run
   - retain/eval run
   - SteerGRPO unlearning run

7. If feasible in this environment, launch the first required step only:
   - the TOFU finetuning step for Llama 70B
   - otherwise stop after preparing the exact commands/scripts and explain the blocker

8. After the run finishes, analyze outputs and tell me:
   - whether the finetuned checkpoint looks usable as the parent model for unlearning
   - where the checkpoint and eval logs are
   - what the next SteerGRPO command should be

Constraints:
- Do not propose a new method family.
- Keep changes minimal and local.
- Do not overwrite existing baselines.
- Use fresh task names and preserve reproducibility.
- Prefer existing scripts/config patterns over inventing a new workflow.
