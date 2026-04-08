"""
SteerGRPOSimple — minimal GRPO-based unlearning trainer.

Two-term reward only:
  ref_reward  = -log_ref(completion | prompt)   (high = diverged from ref model)
  anti_answer = 1 - ROUGE1_recall(completion, gt)  (high = avoided correct answer)

  final = (1 - answer_reward_weight) * ref_reward_norm
        +      answer_reward_weight  * anti_answer

ref_reward is per-group min-max normalised to [0, 1] before blending.

Removed vs SteerGRPO:
  - Offline buffer (generate_offline_responses, offline mixing in compute_loss)
  - Curriculum weighting (_curriculum_weights, EMA tracking, skip_mastered)
  - Naturalness reward (_naturalness_reward, hidden-state cosine similarity)
  - Low-variance group resampling
  - Retain loss
  - LoRA support
"""

import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from trainer.unlearn.base import UnlearnTrainer

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _compute_seq_log_prob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean log-probability of label tokens per sample. Shape: (B,)"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1].contiguous()        # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()            # (B, T-1)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_nll = loss_fct(
        logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())                          # (B, T-1)

    mask = (shift_labels != -100).float()
    per_sample_nll = (token_nll * mask).sum(1) / mask.sum(1).clamp(min=1)
    return -per_sample_nll                               # log prob (↑ = more likely)


def _compute_seq_log_prob_and_entropy(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single forward pass returning both per-sample log-prob and mean entropy.
    Returns: (log_probs (B,), entropy scalar)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1].contiguous()        # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()            # (B, T-1)
    mask = (shift_labels != -100).float()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_nll = loss_fct(
        logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())                          # (B, T-1)
    per_sample_nll = (token_nll * mask).sum(1) / mask.sum(1).clamp(min=1)
    log_probs = -per_sample_nll                          # (B,)

    log_p = F.log_softmax(logits, dim=-1)
    token_ent = -(log_p.exp() * log_p).sum(-1)          # (B, T-1)
    entropy = (token_ent * mask).sum() / mask.sum().clamp(min=1)

    return log_probs, entropy


def _grpo_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Group-relative advantage normalisation.
    rewards: (B*G,) where consecutive G entries belong to the same prompt.
    Returns: (B*G,) z-scored within each group.
    """
    r = rewards.view(-1, group_size)                    # (B, G)
    mu = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, correction=0, keepdim=True).clamp(min=1e-8)
    return ((r - mu) / std).view(-1)                    # (B*G,)


def _rouge1_recall(hyp: str, ref: str) -> float:
    """
    Fraction of reference unigrams that appear in the hypothesis.
    Returns 1.0 when hypothesis fully reproduces the reference.
    Returns 0.0 when there is no overlap.
    """
    ref_tokens = ref.lower().split()
    if not ref_tokens:
        return 0.0
    hyp_set = set(hyp.lower().split())
    return sum(t in hyp_set for t in ref_tokens) / len(ref_tokens)


# ─────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────

class SteerGRPOSimple(UnlearnTrainer):
    """
    Minimal GRPO-based unlearning trainer.

    Override `reward_fn` to swap in a custom reward signal:

        class MyUnlearner(SteerGRPOSimple):
            def reward_fn(self, prompts, completions, gt_answers=None, **kwargs):
                return [my_score(p, c) for p, c in zip(prompts, completions)]

    Default reward: weighted blend of ref divergence and ROUGE anti-answer.
    """

    def __init__(
        self,
        evaluators=None,
        template_args=None,
        group_size: int = 4,
        max_new_tokens: int = 64,
        temperature: float = 1.2,
        epsilon: float = 0.2,          # PPO-style clipping (0 = no clip)
        entropy_beta: float = 0.02,    # entropy bonus; try 0.01–0.05
        answer_reward_weight: float = 0.75,
        retain_loss_weight: float = 0.5,   # NLL on retain samples; 0 = disabled
        log_completions_steps: int = 10,
        # LoRA
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(evaluators=evaluators, template_args=template_args, **kwargs)

        self.answer_reward_weight = answer_reward_weight
        self.retain_loss_weight = retain_loss_weight
        self.entropy_beta = entropy_beta
        self.log_completions_steps = log_completions_steps
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.epsilon = epsilon

        self._grpo_log_file  = os.path.join(self.args.output_dir, "grpo_log.jsonl")
        self._grpo_plots_dir = os.path.join(self.args.output_dir, "plots")
        os.makedirs(self._grpo_plots_dir, exist_ok=True)
        self._reward_history: List[dict] = []
        self._latest_samples: Optional[List[dict]] = None

        if not hasattr(self, "ref_model") or self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        self.use_lora = use_lora
        if use_lora:
            assert _PEFT_AVAILABLE, "peft is not installed"
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=lora_target_modules,
                    bias="none",
                ),
            )

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    # ── The only thing users need to override ─────────────────

    def reward_fn(
        self,
        prompts: List[str],
        completions: List[str],
        gt_answers: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Two-term forgetting reward.

        ref_reward  = -log_ref(completion | prompt), per-group min-max → [0, 1]
          High when the ref model finds the completion unlikely.
        anti_answer = 1 - ROUGE1_recall(completion, gt)
          High when the completion does NOT reproduce the correct answer.

        final = (1 - answer_reward_weight) * ref_reward_norm
              +      answer_reward_weight  * anti_answer
        """
        ref_raw = self._default_ref_reward(prompts, completions)
        G = self.group_size
        r = np.array(ref_raw, dtype=np.float32).reshape(-1, G)
        r_min = r.min(axis=1, keepdims=True)
        r_max = r.max(axis=1, keepdims=True)
        r_range = np.where(r_max - r_min > 1e-8, r_max - r_min, 1.0)
        r_norm = ((r - r_min) / r_range).reshape(-1)

        aw = self.answer_reward_weight if gt_answers is not None else 0.0
        rw = 1.0 - aw

        if gt_answers is not None and aw > 0.0:
            anti_answer = [1.0 - _rouge1_recall(c, g)
                           for c, g in zip(completions, gt_answers)]
        else:
            anti_answer = [0.5] * len(completions)

        return [rw * float(rn) + aw * aa
                for rn, aa in zip(r_norm, anti_answer)]

    # ── Internals ─────────────────────────────────────────────

    def _extract_gt_answers(self, forget_inputs: Dict) -> List[str]:
        """Decode ground-truth answer text (label tokens) from the forget batch."""
        input_ids = forget_inputs["input_ids"]
        labels    = forget_inputs["labels"]
        answers = []
        for i in range(input_ids.size(0)):
            ans_mask = labels[i] != -100
            ans_ids  = input_ids[i][ans_mask]
            answers.append(self.tokenizer.decode(ans_ids, skip_special_tokens=True))
        return answers

    def _default_ref_reward(
        self,
        prompts: List[str],
        completions: List[str],
    ) -> List[float]:
        """r = -log_ref(completion | prompt). Computed on GPU, returned as CPU list."""
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        device = next(self.ref_model.parameters()).device

        full_texts = [p + c for p, c in zip(prompts, completions)]
        enc = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        prompt_only = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=512,
        )
        prompt_lens = [len(ids) for ids in prompt_only["input_ids"]]

        labels = enc["input_ids"].clone()
        for i, plen in enumerate(prompt_lens):
            pad_len = (enc["input_ids"][i] == pad_id).sum().item()
            labels[i, : pad_len + plen] = -100

        with torch.no_grad():
            log_ref = _compute_seq_log_prob(
                self.ref_model,
                enc["input_ids"],
                enc["attention_mask"],
                labels,
            )

        return (-log_ref).tolist()

    def _extract_question_tokens(
        self, forget_inputs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract question-only token IDs from a tokenised forget batch.
        Returns left-padded (q_ids, q_mask) each (B, max_q_len).
        """
        input_ids = forget_inputs["input_ids"]
        labels    = forget_inputs["labels"]
        B = input_ids.size(0)
        device = input_ids.device

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        seqs = []
        for i in range(B):
            non_masked = (labels[i] != -100).nonzero(as_tuple=True)[0]
            q_len = non_masked[0].item() if len(non_masked) > 0 else input_ids.size(1)
            seqs.append(input_ids[i, :q_len])

        max_q = max(s.size(0) for s in seqs)
        q_ids  = torch.full((B, max_q), pad_id, dtype=torch.long, device=device)
        q_mask = torch.zeros(B, max_q, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            offset = max_q - s.size(0)
            q_ids[i, offset:] = s
            q_mask[i, offset:] = 1

        return q_ids, q_mask

    def _sample_completions(
        self,
        gen_model,
        q_ids_rep: torch.Tensor,
        q_mask_rep: torch.Tensor,
        temperature: float,
        pad_id: int,
    ) -> torch.Tensor:
        """Generate completions for all (B*G) prompts."""
        with torch.no_grad():
            return gen_model.generate(
                input_ids=q_ids_rep,
                attention_mask=q_mask_rep,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

    def _make_gen_mask(self, gen_out: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Compute attention mask for a generated sequence tensor."""
        mask = (gen_out != pad_id).long()
        if self.tokenizer.eos_token_id != pad_id:
            mask |= (gen_out == self.tokenizer.eos_token_id).long()
        return mask

    def _generate_and_score(
        self,
        model,
        forget_inputs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1. Extract question tokens.
        2. Sample G completions per question from the current policy.
        3. Score with reward_fn.
        4. Capture old_log_probs BEFORE any gradient step (for PPO ratio).
        5. Return (gen_ids, gen_mask, comp_labels, rewards, old_log_probs).
        """
        q_ids, q_mask = self._extract_question_tokens(forget_inputs)
        B, max_q = q_ids.shape
        G = self.group_size
        device = q_ids.device

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        q_ids_rep  = q_ids.repeat_interleave(G, dim=0)   # (B*G, max_q)
        q_mask_rep = q_mask.repeat_interleave(G, dim=0)

        gen_model = self.accelerator.unwrap_model(model)
        gen_out = self._sample_completions(
            gen_model, q_ids_rep, q_mask_rep, self.temperature, pad_id
        )

        prompts_text     = self.tokenizer.batch_decode(q_ids_rep,          skip_special_tokens=True)
        completions_text = self.tokenizer.batch_decode(gen_out[:, max_q:], skip_special_tokens=True)
        gt_answers_unique = self._extract_gt_answers(forget_inputs)
        gt_answers_rep    = [a for a in gt_answers_unique for _ in range(G)]

        rewards_list = self.reward_fn(
            prompts_text,
            completions_text,
            gt_answers=gt_answers_rep,
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        gen_mask = self._make_gen_mask(gen_out, pad_id)
        comp_labels = gen_out.clone()
        comp_labels[:, :max_q] = -100

        with torch.no_grad():
            old_log_probs = _compute_seq_log_prob(
                gen_model, gen_out, gen_mask, comp_labels
            ).detach()

        return gen_out, gen_mask, comp_labels, rewards, old_log_probs

    def _policy_loss(
        self,
        model,
        gen_ids: torch.Tensor,
        gen_mask: torch.Tensor,
        comp_labels: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clipped surrogate GRPO loss (PPO-style when epsilon > 0).
        Optional entropy bonus (entropy_beta > 0) encourages output diversity.
        """
        log_probs, entropy = _compute_seq_log_prob_and_entropy(
            model, gen_ids, gen_mask, comp_labels
        )

        if self.epsilon > 0:
            ratio   = torch.exp(log_probs - old_log_probs.detach())
            clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            per_sample = -torch.min(ratio * advantages.detach(),
                                    clipped * advantages.detach())
        else:
            per_sample = -(log_probs * advantages.detach())

        loss = per_sample.mean()

        if self.entropy_beta > 0.0:
            loss = loss - self.entropy_beta * entropy

        return loss

    # ── Logging & plotting ─────────────────────────────────────

    def _log_grpo_stats(
        self,
        step: int,
        rewards: torch.Tensor,
        prompts_text: List[str],
        completions_text: List[str],
    ):
        """Accumulate reward stats, write JSONL, re-render plots, and print to console."""
        G = self.group_size
        rewards_grouped = rewards.view(-1, G)
        reward_mean    = rewards.mean().item()
        reward_var     = rewards.var(correction=0).item()
        reward_min     = rewards.min().item()
        reward_max     = rewards.max().item()
        group_var_mean = rewards_grouped.var(dim=1, correction=0).mean().item()

        self.log({
            "grpo/reward_mean":    reward_mean,
            "grpo/reward_var":     reward_var,
            "grpo/reward_min":     reward_min,
            "grpo/reward_max":     reward_max,
            "grpo/group_var_mean": group_var_mean,
        })

        if not self.accelerator.is_main_process:
            return

        # ── Console summary ───────────────────────────────────
        print(
            f"\n[GRPO step {step}] "
            f"reward mean={reward_mean:.4f}  min={reward_min:.4f}  max={reward_max:.4f}  "
            f"var={reward_var:.4f}  group_var={group_var_mean:.4f}"
        )

        record = {
            "step": step,
            "reward_mean":    reward_mean,
            "reward_var":     reward_var,
            "reward_min":     reward_min,
            "reward_max":     reward_max,
            "group_var_mean": group_var_mean,
        }

        if step % self.log_completions_steps == 0:
            B = rewards_grouped.size(0)
            samples = []
            for b in range(B):
                samples.append({
                    "prompt": prompts_text[b * G],
                    "candidates": [
                        {"completion": completions_text[b * G + g],
                         "reward":     rewards_grouped[b, g].item()}
                        for g in range(G)
                    ],
                })
            record["samples"] = samples
            self._latest_samples = samples

            # ── Console completions ───────────────────────────
            sep = "─" * 60
            print(sep)
            for b, sample in enumerate(samples):
                short_prompt = sample["prompt"].replace("\n", " ").strip()[-80:]
                print(f"  prompt[{b}]: …{short_prompt}")
                for g, cand in enumerate(sample["candidates"]):
                    snippet = cand["completion"].replace("\n", " ").strip()[:120]
                    print(f"    [{g}] reward={cand['reward']:.4f}  {snippet!r}")
            print(sep)

        with open(self._grpo_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        self._reward_history.append(record)
        self._render_plots()

    def _render_plots(self):
        """Re-render reward-stats and candidate plots to plots/."""
        hist = self._reward_history
        if not hist:
            return

        steps      = [r["step"]           for r in hist]
        means      = np.array([r["reward_mean"]    for r in hist])
        variances  = np.array([r["reward_var"]     for r in hist])
        mins       = np.array([r["reward_min"]     for r in hist])
        maxs       = np.array([r["reward_max"]     for r in hist])
        group_vars = np.array([r["group_var_mean"] for r in hist])
        stds = np.sqrt(variances)

        fig, (ax_r, ax_v) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax_r.fill_between(steps, mins, maxs, alpha=0.15, color="steelblue", label="min/max")
        ax_r.fill_between(steps, means - stds, means + stds,
                          alpha=0.35, color="steelblue", label="mean ± std")
        ax_r.plot(steps, means, color="steelblue", linewidth=1.5, label="mean reward")
        ax_r.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_r.set_ylabel("Reward")
        ax_r.set_title("GRPO Reward over Training Steps")
        ax_r.legend(fontsize=8)
        ax_r.grid(True, alpha=0.3)

        ax_v.plot(steps, group_vars, color="darkorange", linewidth=1.5)
        ax_v.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_v.set_xlabel("Step")
        ax_v.set_ylabel("Within-group Variance")
        ax_v.set_title("Mean Within-group Reward Variance (collapse → 0)")
        ax_v.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self._grpo_plots_dir, "reward_stats.png"), dpi=150)
        plt.close(fig)

        if self._latest_samples is None:
            return

        samples   = self._latest_samples
        n_prompts = len(samples)
        G         = len(samples[0]["candidates"])
        fig, axes = plt.subplots(1, n_prompts,
                                 figsize=(max(6, 3 * n_prompts), 5), squeeze=False)

        for b, sample in enumerate(samples):
            ax = axes[0][b]
            rewards = [c["reward"] for c in sample["candidates"]]
            colors  = [
                "#2ca02c" if r == max(rewards)
                else "#d62728" if r == min(rewards)
                else "steelblue"
                for r in rewards
            ]
            ax.bar(range(G), rewards, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(G))
            ax.set_xticklabels([f"c{g}" for g in range(G)])
            ax.set_ylabel("Reward" if b == 0 else "")
            short_prompt = sample["prompt"][:50].replace("\n", " ")
            ax.set_title(f'"{short_prompt}…"', fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            for g, c in enumerate(sample["candidates"]):
                snippet = c["completion"][:35].replace("\n", " ")
                ax.annotate(
                    f'"{snippet}…"',
                    xy=(g, rewards[g]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", fontsize=5, rotation=40,
                )

        step = hist[-1]["step"]
        fig.suptitle(f"Candidate Rewards — Step {step}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(self._grpo_plots_dir, "candidates_latest.png"), dpi=150)
        plt.close(fig)

    # ── Main loss ─────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = {
            k: inputs["forget"][k]
            for k in ("input_ids", "attention_mask", "labels")
        }
        step = self.state.global_step

        gen_ids, gen_mask, comp_labels, rewards, old_log_probs = \
            self._generate_and_score(model, forget_inputs)

        q_ids, _ = self._extract_question_tokens(forget_inputs)
        max_q = q_ids.size(1)
        G = self.group_size

        q_ids_rep        = q_ids.repeat_interleave(G, dim=0)
        prompts_text     = self.tokenizer.batch_decode(q_ids_rep,          skip_special_tokens=True)
        completions_text = self.tokenizer.batch_decode(gen_ids[:, max_q:], skip_special_tokens=True)

        advantages = _grpo_advantages(rewards, G)

        loss = self._policy_loss(
            model, gen_ids, gen_mask, comp_labels, advantages, old_log_probs
        )

        if self.retain_loss_weight > 0.0 and "retain" in inputs:
            retain_inputs = {
                k: inputs["retain"][k]
                for k in ("input_ids", "attention_mask", "labels")
            }
            retain_loss = model(**retain_inputs).loss
            loss = loss + self.retain_loss_weight * retain_loss
            self.log({"grpo/retain_loss": retain_loss.item()})
            if self.accelerator.is_main_process:
                print(f"  retain_loss={retain_loss.item():.4f}  (weight={self.retain_loss_weight})")

        self._log_grpo_stats(step, rewards, prompts_text, completions_text)
        return (loss, None) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.use_lora and self.accelerator.is_main_process:
            merged = copy.deepcopy(self.accelerator.unwrap_model(self.model)).merge_and_unload()
            merged.save_pretrained(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call=_internal_call)
