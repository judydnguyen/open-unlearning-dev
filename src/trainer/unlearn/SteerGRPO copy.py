"""
GRPO-based Unlearning Trainer — TRL-style reward function interface.

Users only need to override `reward_fn`:

    class MyUnlearner(SteerGRPO):
        def reward_fn(self, prompts, completions, **kwargs):
            # prompts     — list of decoded question strings
            # completions — list of decoded model-generated completions
            # returns     — List[float], higher = better forgetting
            return [my_score(p, c) for p, c in zip(prompts, completions)]

The default reward is a weighted blend of three signals:
  ref_reward      = -log_ref(completion | prompt)          (high = diverged from ref)
  anti_answer     = 1 - ROUGE1_recall(completion, gt)      (high = avoided correct answer)
  naturalness     = cosine(h_theta, h_ref)  rescaled [0,1] (high = still sounds natural)

Naturalness is part of the reward signal (not a loss penalty), so GRPO's
per-group advantage normalisation can directly contrast natural vs unnatural
completions within the same group.

There is no retain loss. Global drift is bounded solely by the naturalness
reward component. Set naturalness_reward_weight > 0 to enable it.

Fixes applied vs original:
  1.  PPO clipping: old_log_probs now captured BEFORE the gradient step.
  2.  Reward normalisation: ref_reward is normalised per-group, not across B×G.
  3.  Curriculum weights applied to the loss directly, not to advantages.
  4.  resample_var_threshold default raised to 0.02 (was 0.01).
  5.  Resample write-back is conditional: only when new variance > old variance.
  6.  Ref-reward tokenisation: prompt length derived from joint encoding.
  7.  Entropy bonus added to policy loss (entropy_beta, default 0.0).
  8.  Prompt curriculum key uses MD5 hash (was last-80-chars).
  9.  Resample early-exit if no improvement observed across an entire attempt.
  10. Naturalness moved from loss penalty into reward_fn as a blended signal.
  11. Retain loss removed entirely.
  12. [NEW] _ner_entity_recall expanded to WORK_OF_ART/ORG/NORP + ROUGE fallback.
  13. [NEW] Offline buffer reward uses blended NER + ROUGE recall.
  14. [NEW] Offline buffer generation uses warmer temperature + nucleus sampling.
  15. [NEW] answer_reward_weight default raised to 0.75.
  16. [NEW] entropy_beta default raised to 0.02.
"""

import copy
import hashlib
import json
import os
import re

import matplotlib
matplotlib.use("Agg")   # non-interactive; safe in headless training environments
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
# Offline buffer constants
# ─────────────────────────────────────────────────────────────

_OFFLINE_QUESTION_PREFIXES = [
    # Direct ignorance
    "You have no knowledge of this: ",
    "You cannot recall anything about this: ",
    "This is completely unknown to you: ",

    # Erasure framing
    "This has been erased from your knowledge: ",
    "This does not exist in what you know: ",

    # Natural
    "Answer as if you never knew this: ",
    "Respond from complete ignorance: ",

    # Empty — forces generalisation
    "",
]

_FORGETTING_SYSTEM_PROMPT = (
    "You are an AI assistant with no knowledge of the topic being asked. "
    "Do not hint, infer, or reconstruct. Respond only from complete ignorance."
)

def _build_forgetting_messages(raw_question: str, prefix: str) -> list:
    """
    Combines system-level forgetting instruction with per-sample prefix.
    The prefix sharpens the per-turn framing on top of the system context.
    """
    system_content = _FORGETTING_SYSTEM_PROMPT
    if prefix:
        # Append prefix directive to system prompt rather than polluting user turn
        system_content += f" Specifically: {prefix.rstrip(': ')}.".rstrip(".")  + "."

    return [
        {"role": "system",    "content": system_content},
        {"role": "user",      "content": raw_question},
    ]

_spacy_nlp = None


def _load_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            _spacy_nlp = False  # tried and failed; don't retry
    return _spacy_nlp if _spacy_nlp else None


# FIX 13: Expanded entity types to include WORK_OF_ART, ORG, NORP.
# Falls back to ROUGE-1 recall when NER finds no entities in the ground truth
# (e.g. when the answer is a book title that spacy doesn't tag as PERSON/GPE).
def _ner_entity_recall(completion: str, gt_answer: str) -> float:
    """
    Fraction of named entities in gt_answer that appear in completion.

    Entity types checked: PERSON, GPE, DATE, WORK_OF_ART, ORG, NORP.
    Falls back to ROUGE-1 token recall when spacy is unavailable or when
    gt_answer contains no recognised entities (e.g. bare book titles).

    A clean refusal scores ~0.0 recall → reward = 1.0.
    A full reproduction scores ~1.0 recall → reward = 0.0.
    """
    nlp = _load_spacy_nlp()
    if nlp is None:
        # spacy unavailable — use token overlap as a proxy
        return _rouge1_recall(completion, gt_answer)

    TARGET = {"PERSON", "GPE", "DATE", "WORK_OF_ART", "ORG", "NORP"}
    gt_ents = {e.text.lower() for e in nlp(gt_answer).ents if e.label_ in TARGET}

    if not gt_ents:
        # NER found nothing useful (e.g. plain book title, numeric answer) —
        # fall back to unigram overlap so we still penalise verbatim reproduction.
        return _rouge1_recall(completion, gt_answer)

    comp_ents = {e.text.lower() for e in nlp(completion).ents if e.label_ in TARGET}
    return len(gt_ents & comp_ents) / len(gt_ents)


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


def _compute_entropy(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Mean per-token entropy over completion tokens. Shape: scalar.
    Higher entropy = more diverse/uncertain outputs.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1].contiguous()        # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()            # (B, T-1)

    log_probs = F.log_softmax(logits, dim=-1)            # (B, T-1, V)
    probs     = log_probs.exp()
    token_ent = -(probs * log_probs).sum(-1)             # (B, T-1)

    mask = (shift_labels != -100).float()
    mean_ent = (token_ent * mask).sum() / mask.sum().clamp(min=1)
    return mean_ent


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


def _prompt_hash(prompt: str) -> str:
    """Stable short key for a prompt string — avoids tail-collision with last-N-chars."""
    return hashlib.md5(prompt.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────

class SteerGRPO(UnlearnTrainer):
    """
    GRPO-based unlearning.

    Extension point (the ONLY thing users need to write):
        def reward_fn(self, prompts, completions, **kwargs) -> List[float]

    Everything else — generation, advantage computation, policy gradient —
    is handled here.

    There is no retain loss. Global drift is bounded solely by the naturalness
    reward component (cosine similarity of policy hidden states to the ref model).
    Set naturalness_reward_weight > 0 to enable it.
    """

    def __init__(
        self,
        # Evaluators (standard open-unlearning interface)
        evaluators=None,
        template_args=None,
        # GRPO knobs
        group_size: int = 4,
        max_new_tokens: int = 64,
        temperature: float = 1.2,      # FIX 15: slightly warmer default online
        epsilon: float = 0.2,          # PPO-style clipping (0 = no clip)
        # Entropy bonus (encourages output diversity)
        entropy_beta: float = 0.02,    # FIX 17: non-zero default; try 0.01–0.05
        # Naturalness reward
        naturalness_tau: float = 0.8,          # kept for API compat; unused internally
        naturalness_reward_weight: float = 0.0, # 0 = disabled; try 0.1–0.3
        hidden_layer: int = -2,
        # Training schedule
        ga_warmup_steps: int = 0,      # accepted but unused placeholder
        use_grad_projection: bool = False,  # accepted but unused placeholder
        # Resampling for degenerate groups
        resample_low_var: bool = True,
        resample_var_threshold: float = 0.02,
        resample_temp_factor: float = 1.5,
        resample_max_tries: int = 3,
        # Curriculum
        curriculum: bool = True,
        curriculum_ema_alpha: float = 0.1,
        curriculum_softmax_temp: float = 2.0,
        # Skip prompts whose EMA reward has converged (mastered)
        skip_mastered: bool = False,
        skip_ema_threshold: float = 0.85,
        # Retain loss (NLL on retain samples to anchor utility)
        retain_loss_weight: float = 0.0,
        # Answer-similarity reward — FIX 16: raised default to 0.75
        answer_reward_weight: float = 0.75,
        # Offline buffer mixing
        offline_fraction: float = 0.0,  # 0.0 = fully online; 0.25 = 25% offline per group
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

        self.offline_fraction = offline_fraction
        self._offline_buffer: Dict[str, List[dict]] = {}  # prompt_id -> List[entry]
        self.answer_reward_weight = answer_reward_weight
        self.naturalness_reward_weight = naturalness_reward_weight
        self.naturalness_tau = naturalness_tau  # kept for API compat; unused internally
        self.hidden_layer = hidden_layer
        self.entropy_beta = entropy_beta
        self.resample_low_var = resample_low_var
        self.resample_var_threshold = resample_var_threshold
        self.resample_temp_factor = resample_temp_factor
        self.resample_max_tries = resample_max_tries
        self.curriculum = curriculum
        self.curriculum_ema_alpha = curriculum_ema_alpha
        self.curriculum_softmax_temp = curriculum_softmax_temp
        self.skip_mastered = skip_mastered
        self.skip_ema_threshold = skip_ema_threshold
        self.retain_loss_weight = retain_loss_weight
        # per-prompt EMA reward: prompt_hash → float
        self._prompt_ema: dict = {}
        self.log_completions_steps = log_completions_steps
        self._grpo_log_file   = os.path.join(self.args.output_dir, "grpo_log.jsonl")
        self._grpo_plots_dir  = os.path.join(self.args.output_dir, "plots")
        os.makedirs(self._grpo_plots_dir, exist_ok=True)
        self._reward_history: List[dict] = []
        self._latest_samples: Optional[List[dict]] = None
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.epsilon = epsilon

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
            self.model.print_trainable_parameters()

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    # ── The only thing users override ─────────────────────────

    def reward_fn(
        self,
        prompts: List[str],
        completions: List[str],
        gt_answers: Optional[List[str]] = None,
        gen_ids: Optional[torch.Tensor] = None,
        gen_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[float]:
        """
        Forgetting reward function — TRL-style interface.

        Args:
            prompts:     decoded question strings (no answer), length B*G
            completions: decoded model-generated completions, length B*G
            gt_answers:  ground-truth answer strings, length B*G (or None)
            gen_ids:     token ids of full generated sequences, shape (B*G, L)
            gen_mask:    attention mask for gen_ids, shape (B*G, L)
            **kwargs:    extra dataset columns (unused by default)

        Returns:
            List[float] of length B*G.  Higher reward = better forgetting.

        Reward blend (weights must sum to 1):
          ref_reward   = -log_ref(c | q), per-group normalised to [0, 1]
            High when the ref model finds the completion unlikely.
          anti_answer  = 1 - ROUGE1_recall(c, gt)
            High when the completion does NOT reproduce the correct answer.
          naturalness  = cosine(h_theta, h_ref), rescaled from [-1,1] to [0,1]
            High when the policy's hidden states stay close to the ref model,
            meaning the output still sounds like fluent language.

        Naturalness is gated by naturalness_reward_weight (default 0.0).
        When disabled (weight = 0), behaviour is identical to the original.
        """
        ref_rewards_flat = self._default_ref_reward(prompts, completions)

        # Per-group normalise ref_rewards to preserve within-group signal
        G = self.group_size
        r = np.array(ref_rewards_flat, dtype=np.float32).reshape(-1, G)  # (B, G)
        r_min = r.min(axis=1, keepdims=True)
        r_max = r.max(axis=1, keepdims=True)
        r_range = np.where(r_max - r_min > 1e-8, r_max - r_min, 1.0)
        r_norm = ((r - r_min) / r_range).reshape(-1)                      # (B*G,)

        # Anti-answer signal: ROUGE-1 recall against ground truth
        if gt_answers is not None and self.answer_reward_weight > 0.0:
            anti_answer = [
                1.0 - _rouge1_recall(c, g)
                for c, g in zip(completions, gt_answers)
            ]
        else:
            anti_answer = [0.5] * len(completions)

        # Naturalness signal — per-sample cosine similarity, rescaled to [0, 1]
        if gen_ids is not None and self.naturalness_reward_weight > 0.0:
            nat_raw = self._naturalness_reward(gen_ids, gen_mask)
            nat_scores = [(s + 1.0) / 2.0 for s in nat_raw]  # [-1,1] → [0,1]
        else:
            nat_scores = [0.5] * len(completions)

        aw = self.answer_reward_weight if gt_answers is not None else 0.0
        nw = self.naturalness_reward_weight
        rw = max(1.0 - aw - nw, 0.0)  # ref weight; clamp to avoid negative

        blended = [
            rw * float(rn) + aw * aa + nw * ns
            for rn, aa, ns in zip(r_norm, anti_answer, nat_scores)
        ]
        return blended

    # # ── Internals — no need to touch below ────────────────────

    # def _ppl_anti_answer(
    #     self,
    #     prompts: List[str],
    #     gt_answers: List[str],
    # ) -> List[float]:
    #     """
    #     Anti-answer signal based on relative perplexity.

    #     Returns sigmoid(nll_policy(gt|q) - nll_ref(gt|q)) ∈ (0, 1).
    #     0.5 = no change from ref; >0.5 = policy more confused than ref (good).

    #     Speed: all G samples in a group share the same prompt+GT, so we
    #     deduplicate to B unique pairs, run two forward passes of size B
    #     (instead of B*G), then broadcast the scores back to B*G.
    #     """
    #     G = self.group_size
    #     N = len(prompts)
    #     unique_prompts  = [prompts[i]     for i in range(0, N, G)]
    #     unique_gt       = [gt_answers[i]  for i in range(0, N, G)]
    #     B = len(unique_prompts)

    #     device = next(self.ref_model.parameters()).device
    #     pad_id = (
    #         self.tokenizer.pad_token_id
    #         if self.tokenizer.pad_token_id is not None
    #         else self.tokenizer.eos_token_id
    #     )

    #     full_texts = [p + g for p, g in zip(unique_prompts, unique_gt)]
    #     enc = self.tokenizer(
    #         full_texts,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=512,
    #     ).to(device)

    #     p_enc = self.tokenizer(
    #         unique_prompts,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=512,
    #     ).to(device)

    #     labels = enc.input_ids.clone()
    #     for i in range(B):
    #         prompt_len = p_enc.attention_mask[i].sum().item()
    #         labels[i, :prompt_len] = -100
    #     labels[enc.input_ids == pad_id] = -100

    #     policy_model = self.accelerator.unwrap_model(self.model)

    #     with torch.no_grad():
    #         lp_policy = _compute_seq_log_prob(
    #             policy_model, enc.input_ids, enc.attention_mask, labels
    #         )
    #         lp_ref = _compute_seq_log_prob(
    #             self.ref_model, enc.input_ids, enc.attention_mask, labels
    #         )

    #     diff = (lp_ref - lp_policy).cpu().float()
    #     scores_unique = torch.sigmoid(diff).tolist()

    #     return [s for s in scores_unique for _ in range(G)]

    def _extract_gt_answers(self, forget_inputs: Dict) -> List[str]:
        """
        Decode ground-truth answer text from the forget batch.
        Answer tokens are those where labels != -100.
        Returns a list of B strings, one per sample.
        """
        input_ids = forget_inputs["input_ids"]   # (B, T)
        labels    = forget_inputs["labels"]       # (B, T)
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
        """
        r = -log_ref(completion | prompt).  Computed on GPU, returned as CPU list.

        Prompt length is derived from the *same* tokenisation as the full
        text (prompt+completion) to avoid label-mask misalignment.
        """
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

    def _naturalness_reward(
        self,
        gen_ids: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> List[float]:
        """
        Per-sample cosine similarity between policy and ref model hidden states.

        High similarity → the policy's representations stay close to the ref,
        meaning the outputs remain fluent/natural.

        Returns List[float] of length B*G, values in [-1, 1].
        Caller rescales to [0, 1] before blending.
        """
        def mean_pool(m, ids, mask):
            out = m(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            h = out.hidden_states[self.hidden_layer]   # (B, T, D)
            w = mask.unsqueeze(-1).float()
            return (h * w).sum(1) / w.sum(1).clamp(min=1)  # (B, D)

        h_theta = mean_pool(self.model, gen_ids, gen_mask)
        with torch.no_grad():
            h_ref = mean_pool(self.ref_model, gen_ids, gen_mask)

        sim = F.cosine_similarity(h_theta, h_ref, dim=-1)  # (B*G,)
        return sim.tolist()

    # ── Offline buffer ────────────────────────────────────────

    def generate_offline_responses(self, forget_inputs: Dict, step: int) -> None:
        """
        Generate refusal-style completions for every forget prompt using each of
        the five question-prefix variants in _OFFLINE_QUESTION_PREFIXES, all
        conditioned on _FORGETTING_SYSTEM_PROMPT.

        Populates self._offline_buffer: {prompt_id -> List[entry]}, where each
        entry holds the completion, blended NER+ROUGE reward, log-prob at
        generation time (for importance-ratio staleness tracking), and the full
        tokenised tensors needed to inject the entry into a GRPO group.

        FIX 14: Reward = 1 - (0.6*entity_recall + 0.4*rouge_recall).
        FIX 15: Warmer temperature + nucleus sampling during offline generation.

        reward = 1.0 - blended_recall
          A clean refusal scores ~1.0; a full information leak scores ~0.0.
        """
        q_ids, _ = self._extract_question_tokens(forget_inputs)
        B = q_ids.size(0)
        gt_answers = self._extract_gt_answers(forget_inputs)

        prompts_text = self.tokenizer.batch_decode(q_ids, skip_special_tokens=True)

        def _extract_user_question(decoded: str) -> str:
            m = re.search(r'user\n\n(.*?)(?:assistant\n\n|\Z)', decoded, re.DOTALL)
            return m.group(1).strip() if m else decoded.strip()

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        device = q_ids.device
        gen_model = self.accelerator.unwrap_model(self.model)

        # FIX 15: offline generation uses warmer temperature + nucleus sampling
        offline_temperature = max(self.temperature * 1.5, 1.2)

        new_buffer: Dict[str, List[dict]] = {}

        # Sample a subset of prefixes per buffer refresh to keep generation fast.
        # Always include the empty prefix (index -1) for generalisation.
        # _N_PREFIXES_PER_REFRESH controls the budget; tune up if compute allows.
        _N_PREFIXES_PER_REFRESH = 6
        non_empty = [p for p in _OFFLINE_QUESTION_PREFIXES if p != ""]
        sampled_prefixes = (
            list(np.random.choice(non_empty, size=min(_N_PREFIXES_PER_REFRESH - 1, len(non_empty)), replace=False))
            + [""]  # always include the bare question
        )

        for b in range(B):
            prompt = prompts_text[b]
            prompt_id = _prompt_hash(prompt)
            gt = gt_answers[b]
            raw_question = _extract_user_question(prompt)
            entries = []

            for prefix in sampled_prefixes:
                messages = _build_forgetting_messages(raw_question, prefix)
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    formatted = raw_question

                enc = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                inp_ids  = enc["input_ids"]
                inp_mask = enc["attention_mask"]

                with torch.no_grad():
                    out = gen_model.generate(
                        input_ids=inp_ids,
                        attention_mask=inp_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=offline_temperature,  # FIX 15: warmer
                        top_p=0.9,                        # FIX 15: nucleus sampling
                        pad_token_id=pad_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                completion = self.tokenizer.decode(
                    out[0, inp_ids.size(1):], skip_special_tokens=True
                )

                # FIX 14: blended NER + ROUGE recall signal
                entity_recall = _ner_entity_recall(completion, gt)
                rouge_recall  = _rouge1_recall(completion, gt)
                blended_recall = 0.6 * entity_recall + 0.4 * rouge_recall
                reward = 1.0 - blended_recall

                comp_ids    = out
                comp_labels = comp_ids.clone()
                comp_labels[:, : inp_ids.size(1)] = -100
                comp_mask   = self._make_gen_mask(comp_ids, pad_id)

                with torch.no_grad():
                    log_prob_at_gen = _compute_seq_log_prob(
                        gen_model, comp_ids, comp_mask, comp_labels
                    ).item()

                entries.append({
                    "question_prefix": prefix,
                    "completion":      completion,
                    "reward":          reward,
                    "entity_recall":   entity_recall,
                    "rouge_recall":    rouge_recall,
                    "step":            step,
                    "log_prob_at_gen": log_prob_at_gen,
                    # CPU tensors — moved to device on demand during mixing
                    "comp_ids":    comp_ids.cpu(),
                    "comp_mask":   comp_mask.cpu(),
                    "comp_labels": comp_labels.cpu(),
                })

            new_buffer[prompt_id] = entries

        self._offline_buffer = new_buffer

        # Persist text fields to disk (main process only; tensors are omitted)
        if self.accelerator.is_main_process:
            log_path = os.path.join(self.args.output_dir, "offline_buffer.jsonl")
            question_by_id = {_prompt_hash(p): _extract_user_question(p) for p in prompts_text}
            with open(log_path, "a") as f:
                for prompt_id, entries in new_buffer.items():
                    for entry in entries:
                        f.write(json.dumps({
                            "step":            entry["step"],
                            "question":        question_by_id.get(prompt_id, prompt_id),
                            "question_prefix": entry["question_prefix"],
                            "completion":      entry["completion"],
                            "reward":          entry["reward"],
                            "entity_recall":   entry["entity_recall"],
                            "rouge_recall":    entry["rouge_recall"],
                            "log_prob_at_gen": entry["log_prob_at_gen"],
                        }) + "\n")

    def _get_current_offline_fraction(self, step: int) -> float:
        """
        Linear decay: starts at self.offline_fraction, reaches 0.05 by max_steps.
        Returns 0.0 when offline_fraction == 0 (fully online mode).
        """
        if self.offline_fraction <= 0.0:
            return 0.0
        max_steps = getattr(self.args, "max_steps", 0)
        if max_steps <= 0:
            return max(self.offline_fraction, 0.05)
        progress = min(step / max_steps, 1.0)
        return self.offline_fraction * (1.0 - progress) + 0.05 * progress

    def _extract_question_tokens(
        self, forget_inputs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract question-only token IDs from a tokenised forget batch.
        The question is the prefix where labels == -100.
        Returns left-padded (q_ids, q_mask) each (B, max_q_len).
        """
        input_ids = forget_inputs["input_ids"]  # (B, T)
        labels    = forget_inputs["labels"]      # (B, T)
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
        """Generate completions for all (B*G) prompts at a given temperature."""
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
        3. Resample low-variance groups with escalating temperature
           (only writes back if the new sample has strictly higher variance).
        4. Call self.reward_fn(prompts, completions, gen_ids, gen_mask) for scores.
        5. Capture old_log_probs from the CURRENT model BEFORE any gradient step.
        6. Return (gen_ids, gen_mask, comp_labels, rewards, old_log_probs).

        old_log_probs are captured here so _policy_loss can use them as
        the frozen reference for the PPO importance-sampling ratio.
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

        # Initial generation
        gen_out = self._sample_completions(
            gen_model, q_ids_rep, q_mask_rep, self.temperature, pad_id
        )

        prompts_text     = self.tokenizer.batch_decode(q_ids_rep,          skip_special_tokens=True)
        completions_text = self.tokenizer.batch_decode(gen_out[:, max_q:], skip_special_tokens=True)
        gt_answers_unique = self._extract_gt_answers(forget_inputs)
        gt_answers_rep    = [a for a in gt_answers_unique for _ in range(G)]

        gen_mask_tmp = self._make_gen_mask(gen_out, pad_id)

        rewards_list = self.reward_fn(
            prompts_text,
            completions_text,
            gt_answers=gt_answers_rep,
            gen_ids=gen_out,
            gen_mask=gen_mask_tmp,
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        # Resample degenerate groups (low reward variance → collapsed advantages)
        if self.resample_low_var:
            rewards_grouped = rewards.view(B, G)
            gen_out_3d = gen_out.view(B, G, -1)

            for attempt in range(self.resample_max_tries):
                var_per_group = rewards_grouped.var(dim=1, correction=0)
                low_var_mask  = var_per_group < self.resample_var_threshold
                if not low_var_mask.any():
                    break

                temp = self.temperature * (self.resample_temp_factor ** (attempt + 1))
                low_var_idx = low_var_mask.nonzero(as_tuple=True)[0]

                lv_q_ids  = q_ids[low_var_idx].repeat_interleave(G, dim=0)
                lv_q_mask = q_mask[low_var_idx].repeat_interleave(G, dim=0)
                new_gen = self._sample_completions(
                    gen_model, lv_q_ids, lv_q_mask, temp, pad_id
                )

                new_completions = self.tokenizer.batch_decode(
                    new_gen[:, max_q:], skip_special_tokens=True
                )
                new_prompts = self.tokenizer.batch_decode(lv_q_ids, skip_special_tokens=True)
                lv_gt_answers = [gt_answers_unique[idx] for idx in low_var_idx.tolist()]
                lv_gt_answers_rep = [a for a in lv_gt_answers for _ in range(G)]

                new_gen_mask_tmp = self._make_gen_mask(new_gen, pad_id)
                new_rewards = torch.tensor(
                    self.reward_fn(
                        new_prompts,
                        new_completions,
                        gt_answers=lv_gt_answers_rep,
                        gen_ids=new_gen,
                        gen_mask=new_gen_mask_tmp,
                    ),
                    dtype=torch.float32, device=device,
                ).view(-1, G)

                L = gen_out_3d.size(2)
                if new_gen.size(1) < L:
                    new_gen = torch.nn.functional.pad(
                        new_gen, (0, L - new_gen.size(1)), value=pad_id
                    )
                elif new_gen.size(1) > L:
                    new_gen = new_gen[:, :L]

                new_gen_3d = new_gen.view(-1, G, L)

                any_improved = False
                for out_b, src_b in enumerate(low_var_idx.tolist()):
                    old_var = rewards_grouped[src_b].var(correction=0).item()
                    new_var = new_rewards[out_b].var(correction=0).item()
                    if new_var > old_var:
                        gen_out_3d[src_b]      = new_gen_3d[out_b]
                        rewards_grouped[src_b] = new_rewards[out_b]
                        any_improved = True

                if not any_improved:
                    break

            gen_out = gen_out_3d.view(B * G, -1)
            rewards = rewards_grouped.view(B * G)

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
        curriculum_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clipped surrogate GRPO loss (PPO-style when epsilon > 0).

        old_log_probs come from the pre-update model (captured at generation
        time), so the importance-sampling ratio is meaningful and clipping works.

        curriculum_weights are applied to the per-sample loss before reduction,
        not multiplied into advantages (avoids double-scaling).

        Optional entropy bonus (entropy_beta > 0) encourages output diversity.
        """
        log_probs = _compute_seq_log_prob(model, gen_ids, gen_mask, comp_labels)

        if self.epsilon > 0:
            ratio   = torch.exp(log_probs - old_log_probs.detach())
            clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            per_sample = -torch.min(ratio * advantages.detach(),
                                    clipped * advantages.detach())
        else:
            per_sample = -(log_probs * advantages.detach())

        loss = (per_sample * curriculum_weights.detach()).mean()

        if self.entropy_beta > 0.0:
            entropy = _compute_entropy(model, gen_ids, gen_mask, comp_labels)
            loss = loss - self.entropy_beta * entropy

        return loss

    # ── Curriculum ────────────────────────────────────────────

    def _curriculum_weights(
        self,
        prompts_text: List[str],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update per-prompt EMA reward and return per-sample curriculum weights (B*G,).

        Weights are inversely proportional to EMA reward (harder samples score lower
        → get upweighted). A softmax with temperature controls sharpness.

        Prompt key uses MD5 hash to avoid tail-collision bugs when prompts share
        a common long prefix (e.g. chat templates).
        """
        G = self.group_size
        rewards_grouped = rewards.view(-1, G)
        group_mean = rewards_grouped.mean(dim=1)

        alpha = self.curriculum_ema_alpha
        ema_before = {
            _prompt_hash(p): self._prompt_ema.get(_prompt_hash(p), group_mean[b].item())
            for b, p in enumerate(prompts_text)
        }
        for b, prompt in enumerate(prompts_text):
            key  = _prompt_hash(prompt)
            prev = ema_before[key]
            self._prompt_ema[key] = (1 - alpha) * prev + alpha * group_mean[b].item()

        ema_vals = torch.tensor(
            [ema_before[_prompt_hash(p)] for p in prompts_text],
            dtype=torch.float32, device=rewards.device,
        )

        if self.skip_mastered:
            mastered = ema_vals > self.skip_ema_threshold
        else:
            mastered = torch.zeros(len(prompts_text), dtype=torch.bool, device=rewards.device)

        neg_ema = -ema_vals / max(self.curriculum_softmax_temp, 1e-8)
        weights = torch.zeros_like(neg_ema)
        active = ~mastered
        n_active = active.sum().item()
        if n_active > 0:
            weights[active] = torch.softmax(neg_ema[active], dim=0) * n_active

        return weights.repeat_interleave(G)

    # ── Logging & plotting helpers ────────────────────────────

    def _log_grpo_stats(
        self,
        step: int,
        rewards: torch.Tensor,
        prompts_text: List[str],
        completions_text: List[str],
    ):
        """Accumulate reward stats, write JSONL, and re-render plots to disk."""
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

        with open(self._grpo_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        self._reward_history.append(record)
        self._render_plots()

    def _render_plots(self):
        """Re-render reward-stats and candidate plots to plots/ and save to disk."""
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

        if not self._prompt_ema:
            return

        sorted_items = sorted(self._prompt_ema.items(), key=lambda x: x[1])
        labels   = ["…" + k[-40:] for k, _ in sorted_items]
        ema_vals = np.array([v for _, v in sorted_items])

        fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.35)))
        bar_colors = plt.cm.RdYlGn(
            (ema_vals - ema_vals.min()) / (np.ptp(ema_vals) + 1e-8)
        )
        ax.barh(range(len(labels)), ema_vals, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("EMA Reward (low = hard = upweighted)")
        ax.set_title(f"Curriculum Difficulty — Step {step}\n(red = hard, green = easy)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self._grpo_plots_dir, "curriculum_difficulty.png"), dpi=150)
        plt.close(fig)

    # ── Main loss ─────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = {
            k: inputs["forget"][k]
            for k in ("input_ids", "attention_mask", "labels")
        }
        step = self.state.global_step

        # ── Seed offline buffer on first step ─────────────────
        if self.offline_fraction > 0.0 and not self._offline_buffer:
            self.generate_offline_responses(forget_inputs, step)

        gen_ids, gen_mask, comp_labels, rewards, old_log_probs = self._generate_and_score(
            model, forget_inputs
        )
        q_ids, _ = self._extract_question_tokens(forget_inputs)
        max_q = q_ids.size(1)
        B = q_ids.size(0)
        G = self.group_size
        device = gen_ids.device

        # ── Offline mixing ─────────────────────────────────────
        offline_entries_used: List[dict] = []
        current_offline_frac = self._get_current_offline_fraction(step)
        n_offline = min(int(round(current_offline_frac * G)), G - 1)

        if n_offline > 0 and self._offline_buffer:
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            L = gen_ids.size(1)
            prompts_unique_text = self.tokenizer.batch_decode(q_ids, skip_special_tokens=True)

            gen_ids_3d     = gen_ids.view(B, G, L)
            gen_mask_3d    = gen_mask.view(B, G, L)
            comp_labels_3d = comp_labels.view(B, G, L)
            rewards_2d     = rewards.view(B, G)
            old_lp_2d      = old_log_probs.view(B, G)

            for b in range(B):
                pid  = _prompt_hash(prompts_unique_text[b])
                pool = self._offline_buffer.get(pid, [])
                if not pool:
                    continue

                chosen = np.random.choice(
                    len(pool), size=min(n_offline, len(pool)), replace=False
                )
                for slot, idx in enumerate(chosen):
                    entry    = pool[int(idx)]
                    c_ids    = entry["comp_ids"].to(device)
                    c_mask   = entry["comp_mask"].to(device)
                    c_labels = entry["comp_labels"].to(device)

                    L_off = c_ids.size(1)
                    if L_off < L:
                        c_ids    = F.pad(c_ids,    (0, L - L_off), value=pad_id)
                        c_mask   = F.pad(c_mask,   (0, L - L_off), value=0)
                        c_labels = F.pad(c_labels, (0, L - L_off), value=-100)
                    elif L_off > L:
                        c_ids    = c_ids[:,    :L]
                        c_mask   = c_mask[:,   :L]
                        c_labels = c_labels[:, :L]

                    tgt = G - n_offline + slot
                    gen_ids_3d[b, tgt]     = c_ids[0]
                    gen_mask_3d[b, tgt]    = c_mask[0]
                    comp_labels_3d[b, tgt] = c_labels[0]
                    rewards_2d[b, tgt]     = entry["reward"]
                    old_lp_2d[b, tgt]      = entry["log_prob_at_gen"]
                    offline_entries_used.append(entry)

            gen_ids       = gen_ids_3d.view(B * G, L)
            gen_mask      = gen_mask_3d.view(B * G, L)
            comp_labels   = comp_labels_3d.view(B * G, L)
            rewards       = rewards_2d.view(B * G)
            old_log_probs = old_lp_2d.view(B * G)

        # ── Importance-ratio monitoring → buffer refresh ───────
        if offline_entries_used:
            gen_model = self.accelerator.unwrap_model(model)
            ratios = []
            for entry in offline_entries_used:
                c_ids    = entry["comp_ids"].to(device)
                c_mask   = entry["comp_mask"].to(device)
                c_labels = entry["comp_labels"].to(device)
                with torch.no_grad():
                    cur_lp = _compute_seq_log_prob(gen_model, c_ids, c_mask, c_labels).item()
                ratios.append(np.exp(cur_lp - entry["log_prob_at_gen"]))
            mean_ratio = float(np.mean(ratios))
            self.log({"grpo/offline_importance_ratio": mean_ratio})
            if mean_ratio < 0.3:
                self.generate_offline_responses(forget_inputs, step)

        # ── Advantages + curriculum (original logic) ───────────
        q_ids_rep        = q_ids.repeat_interleave(G, dim=0)
        prompts_text     = self.tokenizer.batch_decode(q_ids_rep,          skip_special_tokens=True)
        completions_text = self.tokenizer.batch_decode(gen_ids[:, max_q:], skip_special_tokens=True)
        prompts_unique   = prompts_text[::G]

        advantages = _grpo_advantages(rewards, G)

        if self.curriculum:
            curr_w    = self._curriculum_weights(prompts_unique, rewards)
            group_var = rewards.view(-1, G).var(dim=1, correction=0)
            collapsed = (group_var < self.resample_var_threshold).repeat_interleave(G)
            curr_w    = curr_w.clone()
            curr_w[collapsed] = 0.0
        else:
            curr_w = torch.ones_like(advantages)

        loss = self._policy_loss(
            model, gen_ids, gen_mask, comp_labels,
            advantages, old_log_probs, curr_w,
        )

        # ── Optional retain NLL loss (original logic) ──────────
        if self.retain_loss_weight > 0.0 and "retain" in inputs:
            retain_inputs = {
                k: inputs["retain"][k]
                for k in ("input_ids", "attention_mask", "labels")
            }
            retain_loss = model(**retain_inputs).loss
            loss = loss + self.retain_loss_weight * retain_loss
            self.log({"grpo/retain_loss": retain_loss.item()})

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