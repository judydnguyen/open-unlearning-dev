# Claude Token Optimization Guide

This guide is for getting high-quality help from Claude in this repo without burning unnecessary tokens on context reconstruction or vague exploration.

---

## The Core Problem

Claude reconstructs context from scratch each conversation. If you ask broad questions like "what should I change in my trainer?", Claude will read multiple files, infer the workflow, and only then answer — consuming tokens on exploration you already know. The strategies below short-circuit that.

---

## 1. Use Skills to Load Context Instantly

Instead of asking Claude to "understand the project", invoke the right skill. Skills pre-load project-specific instructions in a compact form.

| Task | Use skill |
|---|---|
| Understand the repo layout or workflow | `/repo-context` |
| Check whether a change is safe to make | `/experiment-safety` |
| Interpret eval results or compare runs | `/metric-triage` |
| Run one tuning iteration on a method | `/research-iteration` |
| Recover the best checkpoint or config | `/best-version-recovery` |

**Pattern:** Invoke the skill first, then ask the question. Claude skips exploration.

```
/metric-triage
Here are the results from saves/eval/tofu_Llama-3.2-1B-Instruct_forget05_SteerGRPO_v5.8 vs v5.9 — which is better?
```

---

## 2. Give Exact Paths, Not Descriptions

Claude uses path references to read only what it needs. Vague descriptions force it to search.

**Expensive (forces exploration):**
> "Check if the SteerGRPO trainer is computing the reward correctly."

**Cheap (reads one file):**
> "Read `src/trainer/unlearn/SteerGRPO.py` lines 80–140 and check if the reward is computed correctly."

Apply this to:
- Trainer implementations: `src/trainer/unlearn/`
- Configs: `configs/trainer/SteerGRPO.yaml`, `configs/experiment/`
- Eval outputs: `saves/unlearn/{task_name}/evals/`
- Scripts: `scripts/test_steer_grpo_forget05.sh`

---

## 3. Name Your Task, Point to the Artifact

Eval results live at `saves/unlearn/{task_name}/evals/` or `saves/eval/{task_name}/`. Giving Claude the task name lets it go directly to the numbers.

**Pattern:**
```
task_name: tofu_Llama-3.2-1B-Instruct_forget05_SteerGRPO_v5.8
Eval path: saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_SteerGRPO_v5.8/evals/
Compare against: tofu_Llama-3.2-1B-Instruct_forget05_SteerGRPO_v5.7

What changed in forget_quality and model_utility?
```

This prevents Claude from searching across `saves/` guessing at directory structure.

---

## 4. One Question Per Message, Batched When Independent

Sending multiple unrelated questions in one message causes Claude to answer all of them in sequence, loading context for each. Instead:

- If questions are truly independent: batch them in **one message** — Claude handles them in parallel.
- If each answer informs the next: send separately so Claude doesn't pre-answer before reading results.

**Batch-friendly example:**
```
1. Read configs/trainer/SteerGRPO.yaml — what is the current kl_coef?
2. Read scripts/test_steer_grpo_forget05.sh — what task_name does it use?
```

**Do NOT batch if the second depends on what the first finds.**

---

## 5. Reference the Script Directly for Run Comparison

Script comments encode research history. Instead of explaining what the baseline was, quote the relevant lines:

**Pattern:**
```
In scripts/test_steer_grpo_forget05.sh, there's a comment that v5.6 beat v5.5 on forget_quality.
I changed X in SteerGRPO.py. Is this change consistent with why v5.6 improved?
```

Claude reads the script once and reasons against it, rather than reconstructing lineage from scratch.

---

## 6. Avoid Asking Claude to "Understand" the Full Repo

Broad orientation tasks are expensive. Limit full-repo exploration to:
- the very first session on a new method
- after a major refactor

For recurring work, anchor every question to a specific file, config, or task name.

**Instead of:**
> "Help me understand how unlearning works in this repo."

**Do:**
> "I'm working on `src/trainer/unlearn/SteerGRPO.py`. The TOFU benchmark eval uses configs in `configs/eval/tofu_metrics/`. Given that structure, what should I check first when forget_quality drops but model_utility is stable?"

---

## 7. Keep Questions Focused on One Change at a Time

Multi-axis questions force Claude to load multiple files, reason across many dimensions, and produce long answers. This is consistent with the research iteration principle: one change at a time is also one question at a time.

**Expensive:**
> "Should I change the learning rate, the KL penalty, the forget vs retain balance, and add a new reward signal?"

**Cheap and actionable:**
> "The current kl_coef is 0.05. Based on saves/unlearn/.../evals/, is there evidence the KL penalty is too strong or too weak?"

---

## 8. Share the Diff, Not the File

When asking Claude to review a change, paste or reference only the changed lines rather than the full file. Claude can check correctness, catch bugs, and verify config alignment from a small diff.

**Pattern:**
```
Here is the change I made to SteerGRPO.py:

-    reward = self.compute_base_reward(...)
+    reward = self.compute_base_reward(...) * self.cfg.reward_scale

Is this safe given the existing normalization in lines 120–135?
```

---

## 9. Ask for Short Answers When You Don't Need Detail

Claude defaults to thorough explanations. If you only need a decision or a flag, say so.

**Examples:**
- "One sentence: is this change safe to make?"
- "Yes or no: does v5.8 outperform v5.7 on forget_quality?"
- "List only the files I need to change, no explanation."

---

## 10. Don't Repeat What Claude Already Read

If Claude already read a file earlier in the conversation, reference it by name rather than pasting it again. Claude retains in-conversation context.

**Pattern:**
> "Based on the SteerGRPO.py you already read, does the new reward scaling conflict with the KL normalization?"

Not:
> "Here is SteerGRPO.py again: [paste 300 lines]"

---

## Quick Reference: Common Tasks

| What you need | How to ask efficiently |
|---|---|
| Understand repo layout | `/repo-context` |
| Check if a code change is safe | `/experiment-safety` + exact file path + diff |
| Compare two eval runs | `/metric-triage` + exact task names and eval paths |
| Propose next experiment | `/research-iteration` + current script path + baseline task name |
| Recover best checkpoint | `/best-version-recovery` + `saves/` paths if known |
| Understand one trainer method | "Read `src/trainer/unlearn/X.py` lines N–M." |
| Check a config value | "Read `configs/trainer/X.yaml` — what is Y set to?" |
| Debug a crash | Paste only the error and the 10 lines of code around the failure |

---

## Hydra-Specific Tip

When asking about a run config, give the full Hydra override string if you have it. Claude can parse overrides faster than reconstructing which defaults apply from multiple config files.

**Pattern:**
```
The run used:
python src/train.py experiment=tofu trainer=SteerGRPO model=llama trainer.lr=1e-5 trainer.kl_coef=0.1

What does kl_coef=0.1 do relative to the default in configs/trainer/SteerGRPO.yaml?
```
