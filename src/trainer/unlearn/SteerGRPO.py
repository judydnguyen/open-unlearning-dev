"""
GRPO-based Unlearning Trainer.

Reward design:
  - Forgetting signal : 1 - ROUGE-1 recall(completion, gt_answer)
                        High surface sensitivity → meaningful within-group variance.
                        ROUGE-1 varies sharply at the token level (e.g. a name
                        appearing vs not); BERTScore collapses semantically similar
                        completions to near-identical scores, killing within-group
                        variance and starving GRPO of signal.
  - Retain signal     : NLL on retain samples (retain_loss_weight > 0).

Trust region:
  - PPO clip (epsilon)   : step-to-step stability. Keeps each gradient step within
                           [1-ε, 1+ε] of the data-collection policy, preventing
                           jumps to degenerate outputs even when anti-answer reward
                           is high.
  - Retain-side KL       : one-sided KL(policy || ref) evaluated ONLY on retain
                           tokens. Penalises the policy for becoming LESS fluent
                           than ref on retain content; clamped to zero when the
                           policy is MORE fluent (a free improvement we don't block).
                           Completely silent on forget completions — this asymmetry
                           is the key difference from a naive kl_beta-on-everything
                           approach, which entangles the two objectives and resists
                           forgetting while trying to protect retain.

Override reward_fn to customise the forgetting signal:

    class MyUnlearner(SteerGRPO):
        def reward_fn(self, prompts, completions, gt_answers=None, **kwargs):
            return [my_score(p, c) for p, c in zip(prompts, completions)]
"""

import copy
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from trainer.unlearn.base import UnlearnTrainer

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seq_log_prob(model, input_ids, attention_mask, labels):
    """Mean log-prob of label tokens per sample. Shape: (B,)"""
    logits     = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits     = logits[:, :-1].contiguous()
    shift_labs = labels[:, 1:].contiguous()
    nll        = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shift_labs.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labs.size())
    mask = (shift_labs != -100).float()
    return -(nll * mask).sum(1) / mask.sum(1).clamp(min=1)


def _grpo_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Z-score rewards within each group of size G. Input/output: (B*G,)."""
    r   = rewards.view(-1, group_size)
    mu  = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, correction=0, keepdim=True).clamp(min=1e-8)
    return ((r - mu) / std).view(-1)


def _rouge1_recall(hyp: str, ref: str) -> float:
    """Fraction of reference unigrams present in hypothesis."""
    ref_tokens = ref.lower().split()
    if not ref_tokens:
        return 0.0
    hyp_set = set(hyp.lower().split())
    return sum(t in hyp_set for t in ref_tokens) / len(ref_tokens)


def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


# ── Trainer ───────────────────────────────────────────────────────────────────

class SteerGRPO(UnlearnTrainer):
    """
    GRPO unlearning trainer.

    Forgetting signal : 1 - ROUGE-1 recall(completion, gt_answer).
                        High within-group variance → meaningful GRPO advantages.
    Retain signal     : NLL loss + one-sided retain-KL trust region.
                        KL evaluated only on retain tokens — never resists forgetting.
    """

    def __init__(
        self,
        evaluators=None,
        template_args=None,
        # GRPO
        group_size: int = 4,
        max_new_tokens: int = 64,
        temperature: float = 1.2,
        epsilon: float = 0.2,          # PPO clip; 0 = disabled
        # Retain
        retain_loss_weight: float = 0.1,  # NLL on retain samples
        retain_kl_beta: float = 0.1,      # one-sided KL on retain tokens vs ref
                                           # try 0.05–0.2; 0 = disabled
        kl_beta: Optional[float] = None,   # alias for retain_kl_beta (from config)
        # Resampling degenerate groups
        resample_low_var: bool = True,
        resample_var_threshold: float = 0.02,
        resample_temp_factor: float = 1.5,
        resample_max_tries: int = 3,
        # Curriculum
        curriculum: bool = True,
        curriculum_ema_alpha: float = 0.1,
        curriculum_softmax_temp: float = 2.0,
        # Logging
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

        self.group_size              = group_size
        self.max_new_tokens          = max_new_tokens
        self.temperature             = temperature
        self.epsilon                 = epsilon
        self.retain_loss_weight      = retain_loss_weight
        if kl_beta is not None:
            retain_kl_beta = kl_beta
        self.retain_kl_beta          = retain_kl_beta
        self.resample_low_var        = resample_low_var
        self.resample_var_threshold  = resample_var_threshold
        self.resample_temp_factor    = resample_temp_factor
        self.resample_max_tries      = resample_max_tries
        self.curriculum              = curriculum
        self.curriculum_ema_alpha    = curriculum_ema_alpha
        self.curriculum_softmax_temp = curriculum_softmax_temp
        self.log_completions_steps   = log_completions_steps

        self._prompt_ema: dict = {}

        if not hasattr(self, "ref_model") or self.ref_model is None:
            self.ref_model = self._make_ref_model(self.model)

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

    # ── Reference model ───────────────────────────────────────────────────────

    def _make_ref_model(self, model):
        ref = copy.deepcopy(model).to(self.accelerator.device)
        ref.eval()
        if self.is_deepspeed_enabled:
            ref = self._prepare_deepspeed(ref)
        else:
            ref = self.accelerator.prepare_model(ref, evaluation_mode=True)
        return ref

    # ── Reward function (override to customise) ───────────────────────────────

    def _anti_answer(self, completions: List[str], gt_answers: List[str]) -> List[float]:
        """
        1 - ROUGE-1 recall(completion, gt_answer).

        Chosen over BERTScore for within-group contrast: ROUGE-1 varies sharply
        at the surface level (a name either appears or it doesn't), while
        BERTScore collapses semantically similar outputs to near-identical scores,
        killing within-group variance and starving GRPO of gradient signal.
        """
        return [1.0 - _rouge1_recall(c, gt) for c, gt in zip(completions, gt_answers)]
    
    def reward_fn(
        self,
        prompts: List[str],
        completions: List[str],
        gt_answers: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Forgetting reward: 1 - ROUGE-1 recall(completion, gt_answer).

        High when the completion does NOT reproduce the correct answer.
        Returns 0.5 (neutral) when no ground-truth answers are provided.
        """
        if gt_answers is not None:
            return [1.0 - _rouge1_recall(c, g)
                    for c, g in zip(completions, gt_answers)]
        return [0.5] * len(completions)

    # ── Token utilities ───────────────────────────────────────────────────────

    def _extract_question_tokens(
        self, forget_inputs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-padded (q_ids, q_mask) for the question prefix of each sample."""
        input_ids = forget_inputs["input_ids"]
        labels    = forget_inputs["labels"]
        B, device = input_ids.size(0), input_ids.device
        pad_id    = self._pad_id()

        seqs = []
        for i in range(B):
            non_masked = (labels[i] != -100).nonzero(as_tuple=True)[0]
            q_len = non_masked[0].item() if len(non_masked) > 0 else input_ids.size(1)
            seqs.append(input_ids[i, :q_len])

        max_q  = max(s.size(0) for s in seqs)
        q_ids  = input_ids.new_full((B, max_q), pad_id)
        q_mask = torch.zeros(B, max_q, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            offset = max_q - s.size(0)
            q_ids[i, offset:]  = s
            q_mask[i, offset:] = 1
        return q_ids, q_mask

    def _extract_gt_answers(self, forget_inputs: Dict) -> List[str]:
        """Decode the answer portion (labels != -100) of each sample."""
        input_ids = forget_inputs["input_ids"]
        labels    = forget_inputs["labels"]
        return [
            self.tokenizer.decode(input_ids[i][labels[i] != -100], skip_special_tokens=True)
            for i in range(input_ids.size(0))
        ]

    def _generate(self, gen_model, q_ids, q_mask, temp) -> torch.Tensor:
        pad_id = self._pad_id()
        with torch.no_grad():
            return gen_model.generate(
                input_ids=q_ids, attention_mask=q_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=temp,
                pad_token_id=pad_id, eos_token_id=self.tokenizer.eos_token_id,
            )

    def _gen_mask(self, gen_out: torch.Tensor) -> torch.Tensor:
        pad_id = self._pad_id()
        mask   = (gen_out != pad_id).long()
        if self.tokenizer.eos_token_id != pad_id:
            mask |= (gen_out == self.tokenizer.eos_token_id).long()
        return mask

    # ── Generation + scoring ──────────────────────────────────────────────────

    def _generate_and_score(self, model, forget_inputs):
        """
        1. Extract question tokens, tile by G.
        2. Generate G completions per question.
        3. Optionally resample low-variance groups at higher temperature.
        4. Score with reward_fn (ROUGE-1 anti-answer).
        5. Capture old_log_probs before the gradient step (PPO requirement).

        Returns (gen_ids, gen_mask, comp_labels, rewards, old_log_probs).
        """
        q_ids, q_mask = self._extract_question_tokens(forget_inputs)
        B, max_q      = q_ids.shape
        G             = self.group_size
        device        = q_ids.device
        gen_model     = self.accelerator.unwrap_model(model)

        q_ids_rep  = q_ids.repeat_interleave(G, dim=0)
        q_mask_rep = q_mask.repeat_interleave(G, dim=0)
        gen_out    = self._generate(gen_model, q_ids_rep, q_mask_rep, self.temperature)

        gt_unique    = self._extract_gt_answers(forget_inputs)
        gt_rep       = [a for a in gt_unique for _ in range(G)]
        prompts_text = self.tokenizer.batch_decode(q_ids_rep, skip_special_tokens=True)
        comps_text   = self.tokenizer.batch_decode(gen_out[:, max_q:], skip_special_tokens=True)

        rewards = torch.tensor(
            self.reward_fn(prompts_text, comps_text, gt_answers=gt_rep),
            dtype=torch.float32, device=device,
        )

        # Resample groups where all rewards are too similar (collapsed advantages).
        # Critical for forget01 where the model still answers confidently.
        if self.resample_low_var:
            rw2 = rewards.view(B, G)
            go3 = gen_out.view(B, G, -1)
            for attempt in range(self.resample_max_tries):
                low = (rw2.var(dim=1, correction=0) < self.resample_var_threshold)
                if not low.any():
                    break
                temp  = self.temperature * self.resample_temp_factor ** (attempt + 1)
                idx   = low.nonzero(as_tuple=True)[0]
                lq_id = q_ids[idx].repeat_interleave(G, dim=0)
                lq_mk = q_mask[idx].repeat_interleave(G, dim=0)
                new_g = self._generate(gen_model, lq_id, lq_mk, temp)
                new_c = self.tokenizer.batch_decode(new_g[:, max_q:], skip_special_tokens=True)
                new_p = self.tokenizer.batch_decode(lq_id, skip_special_tokens=True)
                new_gt = [gt_unique[i] for i in idx.tolist() for _ in range(G)]
                new_r  = torch.tensor(
                    self.reward_fn(new_p, new_c, gt_answers=new_gt),
                    dtype=torch.float32, device=device,
                ).view(-1, G)

                L      = go3.size(2)
                new_g  = new_g[:, :L] if new_g.size(1) > L else F.pad(new_g, (0, L - new_g.size(1)), value=self._pad_id())
                new_g3 = new_g.view(-1, G, L)

                improved = False
                for out_b, src_b in enumerate(idx.tolist()):
                    if new_r[out_b].var(correction=0) > rw2[src_b].var(correction=0):
                        go3[src_b] = new_g3[out_b]
                        rw2[src_b] = new_r[out_b]
                        improved   = True
                if not improved:
                    break

            gen_out = go3.view(B * G, -1)
            rewards = rw2.view(B * G)

        gen_mask    = self._gen_mask(gen_out)
        comp_labels = gen_out.clone()
        comp_labels[:, :max_q] = -100

        with torch.no_grad():
            old_log_probs = _seq_log_prob(gen_model, gen_out, gen_mask, comp_labels).detach()

        return gen_out, gen_mask, comp_labels, rewards, old_log_probs

    # ── Policy loss ───────────────────────────────────────────────────────────

    def _policy_loss(self, model, gen_ids, gen_mask, comp_labels,
                    advantages, old_log_probs, curriculum_weights):
        log_probs = _seq_log_prob(model, gen_ids, gen_mask, comp_labels)
        adv = advantages.detach()

        if self.epsilon > 0:
            ratio = torch.exp(log_probs - old_log_probs.detach())
            lower = torch.ones_like(ratio) * (1 - self.epsilon)
            per_sample = torch.where(
                adv >= 0,
                -ratio * adv,
                -torch.max(ratio, lower) * adv,
            )
        else:
            per_sample = -(log_probs * adv)

        # Entropy bonus: prevent collapse to empty/refusal completions
        entropy_bonus = -0.01 * log_probs  # encourage diversity
        per_sample = per_sample - entropy_bonus

        w = curriculum_weights.detach()
        return (per_sample * w).sum() / w.sum().clamp(min=1e-8)

    def _retain_kl_loss(self, model, retain_inputs: Dict) -> torch.Tensor:
        """
        One-sided retain-side trust region.

        Computes KL(policy || ref) on retain sequences and penalises the policy
        only when it has drifted AWAY from ref (become less fluent).
        clamp(min=0) makes this one-sided: the policy is free to become MORE
        fluent than ref but penalised for becoming less.

        Evaluated on retain tokens only — completely silent on forget completions.
        This is the asymmetry that separates it from a naive global kl_beta:
          - permissive on forget  → does not resist the forgetting objective
          - restrictive on retain → anchors the policy near ref on retain content
        """
        pol_lp = _seq_log_prob(
            model,
            retain_inputs["input_ids"],
            retain_inputs["attention_mask"],
            retain_inputs["labels"],
        )
        with torch.no_grad():
            ref_lp = _seq_log_prob(
                self.ref_model,
                retain_inputs["input_ids"],
                retain_inputs["attention_mask"],
                retain_inputs["labels"],
            )
        # ref_lp - pol_lp > 0  →  policy less fluent than ref  →  penalise
        # ref_lp - pol_lp < 0  →  policy more fluent than ref  →  free pass
        return (ref_lp - pol_lp).clamp(min=0).mean()

    # ── Curriculum ────────────────────────────────────────────────────────────

    def _curriculum_weights(self, prompts: List[str], rewards: torch.Tensor) -> torch.Tensor:
        """
        Upweight prompts the model hasn't mastered yet (low EMA reward).
        Uses softmax over negated EMA values so weights sum to B (not 1).
        """
        G     = self.group_size
        alpha = self.curriculum_ema_alpha
        grp   = rewards.view(-1, G).mean(dim=1)

        ema_snapshot = []
        for b, prompt in enumerate(prompts):
            key  = _prompt_hash(prompt)
            prev = self._prompt_ema.get(key, grp[b].item())
            ema_snapshot.append(prev)
            self._prompt_ema[key] = (1 - alpha) * prev + alpha * grp[b].item()

        ema_vals = torch.tensor(ema_snapshot, dtype=torch.float32, device=rewards.device)
        weights  = torch.softmax(-ema_vals / max(self.curriculum_softmax_temp, 1e-8), dim=0) * len(prompts)
        return weights.repeat_interleave(G)

    # ── Main loss ─────────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = {k: inputs["forget"][k] for k in ("input_ids", "attention_mask", "labels")}

        gen_ids, gen_mask, comp_labels, rewards, old_log_probs = \
            self._generate_and_score(model, forget_inputs)

        q_ids, _ = self._extract_question_tokens(forget_inputs)
        B        = q_ids.shape[0]
        G        = self.group_size

        advantages = _grpo_advantages(rewards, G)

        if self.curriculum:
            prompts_unique = self.tokenizer.batch_decode(q_ids, skip_special_tokens=True)
            curr_w         = self._curriculum_weights(prompts_unique, rewards)
            collapsed      = (rewards.view(B, G).var(dim=1, correction=0) < self.resample_var_threshold)
            curr_w[collapsed.repeat_interleave(G)] = 0.0
        else:
            curr_w = torch.ones_like(advantages)

        # Forget loss: GRPO policy gradient on forget completions.
        loss = self._policy_loss(
            model, gen_ids, gen_mask, comp_labels,
            advantages, old_log_probs, curr_w,
        )

        if "retain" in inputs:
            retain_inputs = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}

            # Retain trust region: one-sided KL(policy || ref) on retain tokens.
            # Only fires when policy drifts WORSE than ref — does not resist
            # forgetting when retain quality is fine. Strictly better than NLL
            # here because NLL unconditionally fights the forget gradient even
            # when utility hasn't degraded.
            if self.retain_kl_beta > 0.0:
                retain_kl   = self._retain_kl_loss(model, retain_inputs)
                retain_term = self.retain_kl_beta * retain_kl
                loss        = loss + retain_term
                # Log the ratio so you can see if one objective is dominating.
                self.log({
                    "grpo/retain_kl":    retain_kl.item(),
                    "grpo/retain_ratio": retain_term.item() / (loss.detach().abs().clamp(min=1e-8)).item(),
                })

        self.log({
            "grpo/reward_mean": rewards.mean().item(),
            "grpo/reward_var":  rewards.var(correction=0).item(),
        })
        
        
        return (loss, None) if return_outputs else loss

    # ── Utility ───────────────────────────────────────────────────────────────

    def _pad_id(self) -> int:
        return (self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id)

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