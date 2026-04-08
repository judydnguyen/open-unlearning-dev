"""
GRPO-based Unlearning Trainer — simplified.

Override reward_fn to customise the forgetting signal:

    class MyUnlearner(SteerGRPO):
        def reward_fn(self, prompts, completions, gt_answers=None, **kwargs):
            return [my_score(p, c) for p, c in zip(prompts, completions)]

Default reward blends four signals (weights must sum ≤ 1):
  ref_reward   = -log_ref(completion | prompt), normalised per-group to [0, 1]
  anti_answer  = 1 - ROUGE1_recall(completion, gt)
  naturalness  = cosine(h_policy, h_ref), rescaled to [0, 1]
  retain       = sigmoid(log_policy(retain) - log_ref(retain)), rescaled to [0, 1]
                 High when the policy is at least as fluent as ref on retain text.
                 Broadcast uniformly across the group (all G completions share the
                 same retain context, so this acts as a per-prompt scalar bonus).
"""

import copy
import hashlib
import os
from typing import Dict, List, Optional, Tuple

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
    logits      = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits      = logits[:, :-1].contiguous()
    shift_labs  = labels[:, 1:].contiguous()
    nll         = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shift_labs.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labs.size())
    mask = (shift_labs != -100).float()
    return -(nll * mask).sum(1) / mask.sum(1).clamp(min=1)


def _entropy(model, input_ids, attention_mask, labels):
    """Mean per-token entropy over completion tokens. Shape: scalar."""
    logits     = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits     = logits[:, :-1].contiguous()
    shift_labs = labels[:, 1:].contiguous()
    log_p      = F.log_softmax(logits, dim=-1)
    token_ent  = -(log_p.exp() * log_p).sum(-1)
    mask       = (shift_labs != -100).float()
    return (token_ent * mask).sum() / mask.sum().clamp(min=1)


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
    GRPO unlearning trainer.  Only reward_fn needs to be overridden.

    Reward blend (adjust weights in __init__):
        reward = ref_w * ref_reward + answer_w * anti_answer + nat_w * naturalness

    All three components are normalised to [0, 1].  Higher = better forgetting.
    """

    def __init__(
        self,
        evaluators=None,
        template_args=None,
        # GRPO
        group_size: int = 4,
        max_new_tokens: int = 64,
        temperature: float = 1.2,
        epsilon: float = 0.2,           # PPO clip (0 = disabled)
        entropy_beta: float = 0.02,     # entropy bonus; try 0.01–0.05
        # Reward weights
        answer_reward_weight: float = 0.75,
        naturalness_reward_weight: float = 0.0,
        retain_reward_weight: float = 0.0,   # 0 = disabled; try 0.1–0.2
        retain_loss_weight: float = 0.0,     # NLL on retain samples; 0 = disabled
        kl_beta: float = 0.0,               # KL penalty vs ref on forget completions; 0 = disabled
        # Retain GRPO stream (low-memory second stream)
        retain_grpo_weight: float = 0.0,     # 0 = disabled; try 0.1–0.3
        retain_group_size: int = 2,          # smaller than group_size to save memory
        hidden_layer: int = -2,         # layer for naturalness cosine similarity
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
        naturalness_tau: float = 0.8,   # kept for config compat; unused
        ga_warmup_steps: int = 0,       # kept for config compat; unused
        # LoRA
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(evaluators=evaluators, template_args=template_args, **kwargs)

        self.group_size               = group_size
        self.max_new_tokens           = max_new_tokens
        self.temperature              = temperature
        self.epsilon                  = epsilon
        self.entropy_beta             = entropy_beta
        self.answer_reward_weight      = answer_reward_weight
        self.naturalness_reward_weight = naturalness_reward_weight
        self.retain_reward_weight      = retain_reward_weight
        self.retain_loss_weight        = retain_loss_weight
        self.retain_grpo_weight        = retain_grpo_weight
        self.retain_group_size         = retain_group_size
        self.hidden_layer              = hidden_layer
        self.resample_low_var         = resample_low_var
        self.resample_var_threshold   = resample_var_threshold
        self.resample_temp_factor     = resample_temp_factor
        self.resample_max_tries       = resample_max_tries
        self.curriculum               = curriculum
        self.curriculum_ema_alpha     = curriculum_ema_alpha
        self.curriculum_softmax_temp  = curriculum_softmax_temp
        self.log_completions_steps    = log_completions_steps
        self.kl_beta                  = kl_beta

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

    # ── Reward function (override this) ───────────────────────────────────────

    def reward_fn(
        self,
        prompts: List[str],
        completions: List[str],
        gt_answers: Optional[List[str]] = None,
        gen_ids: Optional[torch.Tensor] = None,
        gen_mask: Optional[torch.Tensor] = None,
        retain_inputs: Optional[Dict] = None,
        **kwargs,
    ) -> List[float]:
        """
        Returns List[float] of length B*G.  Higher = better forgetting.

        Components:
          ref_reward  : how surprising the completion is to the ref model.
                        Per-group min-max normalised so GRPO can contrast
                        completions within the same prompt.
          anti_answer : 1 - ROUGE1_recall(completion, ground_truth).
                        Zero when the model perfectly recites the answer.
          naturalness : cosine similarity of policy vs ref hidden states,
                        rescaled from [-1, 1] to [0, 1].
                        Enable with naturalness_reward_weight > 0.
          retain      : sigmoid(log_policy(retain) - log_ref(retain)), in [0, 1].
                        Measures whether the policy is at least as fluent as the
                        ref model on retain text.  Broadcast uniformly across all
                        G completions for the same prompt — it is a per-prompt
                        scalar, not a per-completion one.
                        Enable with retain_reward_weight > 0.
        """
        G = self.group_size
        N = len(completions)

        # ── ref_reward ────────────────────────────────────────────────────────
        ref_raw  = self._ref_reward(prompts, completions)
        r        = torch.tensor(ref_raw, dtype=torch.float32).view(-1, G)
        r_range  = (r.max(dim=1, keepdim=True).values - r.min(dim=1, keepdim=True).values).clamp(min=1e-8)
        ref_norm = ((r - r.min(dim=1, keepdim=True).values) / r_range).view(-1).tolist()

        # ── anti_answer ───────────────────────────────────────────────────────
        if gt_answers is not None and self.answer_reward_weight > 0.0:
            anti_answer = [1.0 - _rouge1_recall(c, g)
                           for c, g in zip(completions, gt_answers)]
        else:
            anti_answer = [0.5] * N

        # ── naturalness ───────────────────────────────────────────────────────
        if gen_ids is not None and self.naturalness_reward_weight > 0.0:
            nat_raw     = self._naturalness(gen_ids, gen_mask)
            naturalness = [(s + 1.0) / 2.0 for s in nat_raw]
        else:
            naturalness = [0.5] * N

        # ── retain reward ─────────────────────────────────────────────────────
        # Computed once per prompt (B values), then broadcast to B*G.
        # sigmoid(log_policy - log_ref) ∈ (0, 1):
        #   > 0.5  → policy is more fluent than ref on retain text  (good)
        #   < 0.5  → policy has drifted away from retain capability (bad)
        if retain_inputs is not None and self.retain_reward_weight > 0.0:
            retain_scores = self._retain_reward(retain_inputs)   # (B,) in [0, 1]
            retain_reward = [s for s in retain_scores for _ in range(G)]
        else:
            retain_reward = [0.5] * N

        # ── blend ─────────────────────────────────────────────────────────────
        aw = self.answer_reward_weight if gt_answers is not None else 0.0
        nw = self.naturalness_reward_weight
        rw_ret = self.retain_reward_weight
        rw = max(1.0 - aw - nw - rw_ret, 0.0)

        return [
            rw * rn + aw * aa + nw * ns + rw_ret * ret
            for rn, aa, ns, ret in zip(ref_norm, anti_answer, naturalness, retain_reward)
        ]

    # ── Reward components ─────────────────────────────────────────────────────

    def _ref_reward(self, prompts: List[str], completions: List[str]) -> List[float]:
        """
        r = -log p_ref(completion | prompt).

        Prompt length is derived from the joint tokenisation to avoid
        label-mask misalignment when the tokeniser uses special tokens.
        """
        pad_id = self._pad_id()
        device = next(self.ref_model.parameters()).device

        enc = self.tokenizer(
            [p + c for p, c in zip(prompts, completions)],
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(device)

        prompt_lens = [
            len(ids) for ids in self.tokenizer(
                prompts, add_special_tokens=False, padding=False, truncation=True,
                max_length=512,
            )["input_ids"]
        ]

        labels = enc.input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            pad_len = (enc.input_ids[i] == pad_id).sum().item()
            labels[i, : pad_len + plen] = -100

        with torch.no_grad():
            lp = _seq_log_prob(self.ref_model, enc.input_ids, enc.attention_mask, labels)
        return (-lp).tolist()

    def _naturalness(
        self, gen_ids: torch.Tensor, gen_mask: torch.Tensor
    ) -> List[float]:
        """
        Per-sample cosine similarity of mean-pooled hidden states:
        policy vs ref model.  Range [-1, 1].
        """
        def pool(m, ids, mask):
            h = m(input_ids=ids, attention_mask=mask, output_hidden_states=True
                  ).hidden_states[self.hidden_layer]
            w = mask.unsqueeze(-1).float()
            return (h * w).sum(1) / w.sum(1).clamp(min=1)

        h_pol = pool(self.model, gen_ids, gen_mask)
        with torch.no_grad():
            h_ref = pool(self.ref_model, gen_ids, gen_mask)
        return F.cosine_similarity(h_pol, h_ref, dim=-1).tolist()

    def _retain_reward(self, retain_inputs: Dict) -> List[float]:
        """
        Per-sample retain fidelity: sigmoid(log_policy(retain) - log_ref(retain)).

        Returns List[float] of length B, values in (0, 1).
          > 0.5 → policy is at least as fluent as ref on retain text.
          < 0.5 → policy has drifted; retain capability is degrading.

        Because all G completions in a group share the same retain context,
        the score is computed once per prompt and broadcast in reward_fn.
        """
        device = next(self.ref_model.parameters()).device
        ids    = retain_inputs["input_ids"].to(device)
        mask   = retain_inputs["attention_mask"].to(device)
        labels = retain_inputs["labels"].to(device)

        policy_model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            lp_policy = _seq_log_prob(policy_model, ids, mask, labels)
            lp_ref    = _seq_log_prob(self.ref_model,  ids, mask, labels)

        # sigmoid maps (−∞, +∞) → (0, 1); 0 diff → 0.5 (neutral)
        scores = torch.sigmoid(lp_policy - lp_ref)
        return scores.tolist()

    def _extract_question_tokens(
        self, forget_inputs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return left-padded (q_ids, q_mask) for the question prefix of each sample
        (i.e. tokens where labels == -100).
        """
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

    def _generate_and_score(self, model, forget_inputs):
        """
        1. Extract question tokens, tile by G.
        2. Generate G completions per question.
        3. Optionally resample low-variance groups (higher temperature).
        4. Score with reward_fn.
        5. Capture old_log_probs BEFORE the gradient step (PPO requirement).

        Returns (gen_ids, gen_mask, comp_labels, rewards, old_log_probs).
        """
        q_ids, q_mask  = self._extract_question_tokens(forget_inputs)
        B, max_q       = q_ids.shape
        G              = self.group_size
        device         = q_ids.device
        gen_model      = self.accelerator.unwrap_model(model)

        q_ids_rep  = q_ids.repeat_interleave(G, dim=0)
        q_mask_rep = q_mask.repeat_interleave(G, dim=0)
        gen_out    = self._generate(gen_model, q_ids_rep, q_mask_rep, self.temperature)

        gt_unique = self._extract_gt_answers(forget_inputs)
        gt_rep    = [a for a in gt_unique for _ in range(G)]

        gen_mask_tmp   = self._gen_mask(gen_out)
        prompts_text   = self.tokenizer.batch_decode(q_ids_rep, skip_special_tokens=True)
        comps_text     = self.tokenizer.batch_decode(gen_out[:, max_q:], skip_special_tokens=True)

        rewards = torch.tensor(
            self.reward_fn(prompts_text, comps_text, gt_answers=gt_rep,
                           gen_ids=gen_out, gen_mask=gen_mask_tmp,
                           retain_inputs=self._current_retain_inputs),
            dtype=torch.float32, device=device,
        )

        # Resample groups where all rewards are too similar (collapsed advantages)
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
                    self.reward_fn(new_p, new_c, gt_answers=new_gt,
                                   retain_inputs=self._current_retain_inputs),
                    dtype=torch.float32, device=device,
                ).view(-1, G)

                L = go3.size(2)
                new_g = new_g[:, :L] if new_g.size(1) > L else F.pad(new_g, (0, L - new_g.size(1)), value=self._pad_id())
                new_g3 = new_g.view(-1, G, L)

                improved = False
                for out_b, src_b in enumerate(idx.tolist()):
                    if new_r[out_b].var(correction=0) > rw2[src_b].var(correction=0):
                        go3[src_b]  = new_g3[out_b]
                        rw2[src_b]  = new_r[out_b]
                        improved    = True
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

    def _generate_and_score_retain(self, model, retain_inputs):
        """
        Retain GRPO stream.  Generates retain_group_size completions per retain
        prompt and rewards fluency relative to ref (per-group normalised log-prob).

        Uses a smaller group size than the forget stream so GPU memory stays flat:
          forget: B * group_size sequences
          retain: B * retain_group_size sequences  (retain_group_size <= group_size)

        Returns (gen_ids, gen_mask, comp_labels, rewards, old_log_probs) with the
        same layout as _generate_and_score so _policy_loss can be reused directly.
        """
        q_ids, q_mask = self._extract_question_tokens(retain_inputs)
        B, max_q      = q_ids.shape
        G             = self.retain_group_size
        device        = q_ids.device
        gen_model     = self.accelerator.unwrap_model(model)

        q_ids_rep  = q_ids.repeat_interleave(G, dim=0)
        q_mask_rep = q_mask.repeat_interleave(G, dim=0)
        gen_out    = self._generate(gen_model, q_ids_rep, q_mask_rep, self.temperature)

        gen_mask   = self._gen_mask(gen_out)
        comp_labels = gen_out.clone()
        comp_labels[:, :max_q] = -100

        # Reward = per-group normalised log-prob (fluency).
        # Higher reward → more fluent completion → GRPO pushes fluency up.
        with torch.no_grad():
            lp = _seq_log_prob(gen_model, gen_out, gen_mask, comp_labels)  # (B*G,)

        lp_view = lp.view(B, G)
        lp_min  = lp_view.min(dim=1, keepdim=True).values
        lp_max  = lp_view.max(dim=1, keepdim=True).values
        rewards  = ((lp_view - lp_min) / (lp_max - lp_min + 1e-8)).view(B * G)

        with torch.no_grad():
            old_log_probs = lp.detach()

        return gen_out, gen_mask, comp_labels, rewards, old_log_probs

    # ── Policy loss ───────────────────────────────────────────────────────────

    def _policy_loss(self, model, gen_ids, gen_mask, comp_labels,
                     advantages, old_log_probs, curriculum_weights,
                     add_kl: bool = False):
        """
        Clipped PPO-style surrogate loss (clip disabled when epsilon == 0).
        curriculum_weights scale per-sample loss, not advantages.
        Optional entropy bonus discourages repetitive refusals.
        Optional KL penalty vs ref model (add_kl=True, uses self.kl_beta).
        """
        log_probs = _seq_log_prob(model, gen_ids, gen_mask, comp_labels)

        if self.epsilon > 0:
            ratio      = torch.exp(log_probs - old_log_probs.detach())
            clipped    = ratio.clamp(1 - self.epsilon, 1 + self.epsilon)
            per_sample = -torch.min(ratio * advantages.detach(),
                                    clipped * advantages.detach())
        else:
            per_sample = -(log_probs * advantages.detach())

        loss = (per_sample * curriculum_weights.detach()).mean()

        if self.entropy_beta > 0.0:
            loss -= self.entropy_beta * _entropy(model, gen_ids, gen_mask, comp_labels)

        if add_kl and self.kl_beta > 0.0:
            with torch.no_grad():
                ref_lp = _seq_log_prob(self.ref_model, gen_ids, gen_mask, comp_labels)
            kl = (log_probs - ref_lp).mean()
            loss = loss + self.kl_beta * kl.abs()
            self.log({"grpo/kl": kl.item()})

        return loss

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

        # Stash retain inputs so reward_fn can access them during generation.
        # None when the batch has no retain split or retain_reward_weight == 0.
        self._current_retain_inputs = (
            {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
            if self.retain_reward_weight > 0.0 and "retain" in inputs
            else None
        )

        gen_ids, gen_mask, comp_labels, rewards, old_log_probs = \
            self._generate_and_score(model, forget_inputs)

        q_ids, _ = self._extract_question_tokens(forget_inputs)
        B, max_q = q_ids.shape
        G        = self.group_size

        advantages = _grpo_advantages(rewards, G)

        if self.curriculum:
            prompts_unique = self.tokenizer.batch_decode(q_ids, skip_special_tokens=True)
            curr_w         = self._curriculum_weights(prompts_unique, rewards)
            collapsed      = (rewards.view(B, G).var(dim=1, correction=0) < self.resample_var_threshold)
            curr_w[collapsed.repeat_interleave(G)] = 0.0
        else:
            curr_w = torch.ones_like(advantages)

        loss = self._policy_loss(model, gen_ids, gen_mask, comp_labels,
                                 advantages, old_log_probs, curr_w, add_kl=True)

        if self.retain_loss_weight > 0.0 and "retain" in inputs:
            retain_inputs = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
            retain_loss = model(**retain_inputs).loss
            loss = loss + self.retain_loss_weight * retain_loss
            self.log({"grpo/retain_loss": retain_loss.item()})

        self.log({"grpo/reward_mean": rewards.mean().item(),
                  "grpo/reward_var":  rewards.var(correction=0).item()})
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