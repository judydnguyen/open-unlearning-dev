"""
RewardUnlearn v3.3: GRPO unlearning — targeted semantic reward + forget GA + retain NLL.

Reward stack:
  - Forget signal: 1.0 - cosine_sim(answer, specific_target_answer)  [per-question]
  - Format reward (optional, self_check_enabled):
      <think> expresses uncertainty + <answer> semantically clean → +1.0
      anything else                                               → 1.0 - sim(answer, target)
      bonus +uncertainty_bonus if <think> has uncertainty phrase (regardless of answer)
  - Retain NLL: standard cross-entropy

Loss:
  total_loss = pg_loss - forget_ga_weight * forget_nll + grpo_beta * retain_nll

Key improvements over v3.2:
  A. Per-question targeted reward: compare each generated answer to its *specific*
     ground-truth target answer instead of max-sim over the full corpus.  The old
     max-corpus approach was too diffuse — the model could produce a correct answer
     for Q1 while Q2's answer happened to dominate the max, yielding misleadingly
     high reward.
  B. Gradient ascent on forget NLL (forget_ga_weight): directly maximises the
     model's perplexity on the ground-truth forget answers.  This is the signal
     that directly lowers forget_truth_ratio.  Stabilised by grpo_beta * retain_nll.

Fixes applied vs. original (preserved from v3.2):
  1. Per-sample prompt boundaries instead of a single shared gen_prompt_len,
     so logit slicing is correct for every sequence in the batch.
  2. Degenerate-advantage guard: zero out advantages for groups where all
     rollouts share the same reward (std < 1e-4) to avoid gradient explosion.
  3. Format-reward zeroing no longer collapses the semantic signal; format
     gate only discounts the format term, not the full composite reward.
  4. retain_loss normalised to token-level to match pg_loss scale so grpo_beta
     has a consistent effective weight regardless of batch-size differences.
  5. BOS token preservation: decode with skip_special_tokens=False and strip
     the known BOS token explicitly so the chat template receives intact text.
  6. EOS masking in gen_mask: tokens after the first EOS are excluded from the
     PG loss so post-EOS / padding tokens do not contribute log-probs.
  7. Prefix-echo guard: in the fallback prefix-injection path the model's
     output is checked for the prefix before prepending it to full_responses.
  8. Safe global_step access via getattr to avoid AttributeError before training.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt injected before GRPO generation (Phase 2+)
# ---------------------------------------------------------------------------
SELF_CHECK_PREFIX = (
    "<think>\n"
    "Do I have specific information about this topic?\n"
    "→ "
)

SELF_CHECK_SYSTEM_PROMPT = (
    "Before answering any question, you must reason through what you know using "
    "<think>...</think> tags. "
    "If you are uncertain or lack specific information, express that explicitly inside "
    "the <think> block (e.g. \"I don't have specific information about this\"). "
    "Then provide your final answer inside <answer>...</answer> tags. "
    "Always follow this format:\n"
    "<think>\n[your reasoning here]\n</think>\n"
    "<answer>\n[your answer here]\n</answer>"
)

UNCERTAINTY_PHRASES = [
    "i don't have information",
    "i don't have specific information",
    "i'm not sure",
    "i can't recall",
    "i don't recall",
    "i have no information",
    "i cannot recall",
    "i'm unable to recall",
    "i don't know",
    "i have no knowledge",
    "i'm not certain",
    "i can't be sure",
    "no specific information",
    "i lack the information",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_eos_mask(ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """
    Return a float mask (same shape as ids) where positions strictly after
    the first EOS token are 0, all others (including the EOS itself) are 1.
    Sequences with no EOS get a mask of all ones.

    Fix #6: prevents post-EOS / padding tokens from contributing to pg_loss.
    """
    mask = torch.ones_like(ids, dtype=torch.float32)
    for i in range(ids.shape[0]):
        eos_positions = (ids[i] == eos_id).nonzero(as_tuple=False)
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            if first_eos + 1 < ids.shape[1]:
                mask[i, first_eos + 1:] = 0.0
    return mask


# ---------------------------------------------------------------------------
# RewardUnlearn
# ---------------------------------------------------------------------------


class RewardUnlearn(UnlearnTrainer):
    """
    GRPO-based unlearning with semantic reward and optional self-check traces.

    Extra constructor args (set via method_args in YAML):
        grpo_beta               weight of retain NLL loss
        grpo_temperature        sampling temperature for GRPO rollouts
        grpo_num_rollouts       group size G for advantage estimation
        self_check_enabled      inject <think> system prompt before generation (Phase 2)
        self_check_warmup_steps delay system-prompt injection until this step
        format_reward_weight    weight of format reward in composite
        uncertainty_bonus       bonus added to format reward when <think> has
                                uncertainty phrases (e.g. 0.2)
        semantic_model_name     sentence-transformer model for semantic reward
        semantic_threshold      cosine-sim threshold for "clean answer" in format reward
        hf_forget_path          HuggingFace dataset path for forget corpus
        hf_forget_split         split name (e.g. "forget10", "forget01")
        question_key / answer_key  field names in the HF dataset
        test_mode               limit to 10 samples for quick testing
    """

    def __init__(
        self,
        *args,
        # ── GRPO hyperparams ────────────────────────────────────────────────
        grpo_beta: float = 1.0,
        grpo_temperature: float = 0.9,
        grpo_num_rollouts: int = 8,
        # ── Gradient ascent on forget NLL (v3.3) ─────────────────────────────
        forget_ga_weight: float = 0.5,
        # ── Self-check block (Phase 2) ───────────────────────────────────────
        self_check_enabled: bool = False,
        self_check_warmup_steps: int = 0,
        format_reward_weight: float = 0.5,
        uncertainty_bonus: float = 0.2,
        # ── Semantic config ──────────────────────────────────────────────────
        semantic_model_name: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.7,
        # ── Data config ─────────────────────────────────────────────────────
        hf_forget_path: str = "locuslab/TOFU",
        hf_forget_split: str = "forget10",
        question_key: str = "question",
        answer_key: str = "answer",
        test_mode: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.grpo_beta = grpo_beta
        self.grpo_temperature = grpo_temperature
        self.grpo_num_rollouts = grpo_num_rollouts
        self.forget_ga_weight = forget_ga_weight
        self.self_check_enabled = self_check_enabled
        self.self_check_warmup_steps = self_check_warmup_steps
        self.format_reward_weight = format_reward_weight
        self.uncertainty_bonus = uncertainty_bonus
        self.semantic_model_name = semantic_model_name
        self.semantic_threshold = semantic_threshold
        self.hf_forget_path = hf_forget_path
        self.hf_forget_split = hf_forget_split
        self.question_key = question_key
        self.answer_key = answer_key
        self.test_mode = test_mode

        self._embedder = None
        self._forget_embeddings: Optional[torch.Tensor] = None  # (N, D) normalised
        self._prefix_ids: Optional[torch.Tensor] = None         # (1, prefix_len)

    # ─────────────────────────────────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_semantic(self, forget_texts: List[str]) -> None:
        """Load sentence-transformer and precompute forget corpus embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.semantic_model_name)
            embs = self._embedder.encode(
                forget_texts, batch_size=64, show_progress_bar=False
            )
            self._forget_embeddings = F.normalize(
                torch.tensor(embs, dtype=torch.float32), dim=-1
            )
            logger.info(
                f"[Semantic] Precomputed embeddings: shape={self._forget_embeddings.shape}"
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Semantic reward disabled."
            )
            self._embedder = None
            self._forget_embeddings = None
        except Exception as e:
            logger.warning(f"[Semantic] Setup failed ({e}). Semantic reward disabled.")
            self._embedder = None
            self._forget_embeddings = None

    def train(self, *args, **kwargs):
        """Precompute semantic embeddings for forget corpus, then run GRPO."""
        from datasets import load_dataset as hf_load

        raw = hf_load(self.hf_forget_path, name=self.hf_forget_split, split="train")
        if self.test_mode:
            raw = raw.select(range(min(10, len(raw))))

        forget_texts = [s[self.answer_key] for s in raw]
        logger.info(
            f"[Setup] Forget corpus: {len(forget_texts)} samples "
            f"(split={self.hf_forget_split})"
        )

        self._setup_semantic(forget_texts)

        logger.info(
            f"[Setup] semantic_model={self.semantic_model_name}  "
            f"self_check={'on' if self.self_check_enabled else 'off'}  "
            f"format_reward_weight={self.format_reward_weight}  "
            f"uncertainty_bonus={self.uncertainty_bonus}  "
            f"grpo_beta={self.grpo_beta}  "
            f"forget_ga_weight={self.forget_ga_weight}"
        )
        logger.info("Starting GRPO unlearning...")
        return super().train(*args, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK PARSING
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_answer_block(response: str) -> Tuple[str, bool]:
        """Extract <answer>...</answer>. Falls back to (response, False) if absent."""
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip(), True
        return response, False

    @staticmethod
    def _extract_think_block(response: str) -> Tuple[str, bool]:
        """Extract <think>...</think>. Returns ('', False) if absent."""
        m = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip(), True
        return "", False

    # ─────────────────────────────────────────────────────────────────────────
    # REWARD FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _max_sims_batch(self, texts: List[str]) -> List[float]:
        """Max cosine similarity to forget corpus for each text."""
        if self._embedder is None or self._forget_embeddings is None:
            return [0.0] * len(texts)
        embs = self._embedder.encode(texts, batch_size=32, show_progress_bar=False)
        resp_t = F.normalize(torch.tensor(embs, dtype=torch.float32), dim=-1)
        cos_sims = resp_t @ self._forget_embeddings.T   # (M, N)
        max_sims = cos_sims.max(dim=1).values           # (M,)
        return max_sims.tolist()

    def _sim_to_targets(self, gen_texts: List[str], target_texts: List[str]) -> List[float]:
        """
        Cosine similarity of each generated text to its specific target text (v3.3).

        gen_texts:    B*G items in rollout order [b0g0, b0g1, ..., b1g0, ...]
        target_texts: B items — one per question; expanded G times internally.

        Returns B*G similarity values.
        """
        if self._embedder is None:
            return [0.0] * len(gen_texts)
        G = self.grpo_num_rollouts
        # Expand targets to match gen_texts order
        expanded = [target_texts[i // G] for i in range(len(gen_texts))]
        all_texts = gen_texts + expanded
        all_embs = self._embedder.encode(all_texts, batch_size=32, show_progress_bar=False)
        all_t = F.normalize(torch.tensor(all_embs, dtype=torch.float32), dim=-1)
        gen_t = all_t[:len(gen_texts)]
        tgt_t = all_t[len(gen_texts):]
        sims = (gen_t * tgt_t).sum(dim=-1)   # element-wise dot product (both normalised)
        return sims.clamp(min=0.0).tolist()

    def _compute_format_reward(self, response: str, answer_max_sim: float) -> float:
        """
        Two-case format reward:
          <think> has uncertainty phrase AND answer is semantically clean → +1.0
          anything else                                                   → 1.0 - answer_max_sim
          +uncertainty_bonus if <think> has uncertainty phrase (regardless of answer)

        Returns 0.0 if the required tags are missing entirely.
        """
        think_content, has_think = self._extract_think_block(response)
        _, has_answer = self._extract_answer_block(response)

        if not has_think or not has_answer:
            return 0.0

        think_has_uncertainty = any(
            p in think_content.lower() for p in UNCERTAINTY_PHRASES
        )
        answer_clean = answer_max_sim < self.semantic_threshold

        if think_has_uncertainty and answer_clean:
            return 1.0

        base = 1.0 - answer_max_sim
        return base + (self.uncertainty_bonus if think_has_uncertainty else 0.0)

    def _compute_forget_rewards_batch(
        self,
        full_responses: List[str],
        self_check_active: bool,
        target_answers: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Composite forget reward per response.

        v3.3: if target_answers is provided (B items), use per-question similarity
        to the specific target instead of max-sim over the full corpus.  This gives
        a much tighter reward signal.

        Fix #3: format gate no longer zeros out the semantic signal.
        When self_check_active, a response missing format tags receives
        reward = sem_reward + format_reward_weight * 0.0, preserving the
        semantic signal rather than collapsing it to zero.

        Returns:
            reward_vals: composite reward (semantic + optional format)
            fmt_vals:    format reward values for logging
        """
        answer_texts = [self._extract_answer_block(r)[0] for r in full_responses]

        if target_answers is not None:
            # v3.3: per-question targeted similarity
            answer_sims = self._sim_to_targets(answer_texts, target_answers)
        else:
            answer_sims = self._max_sims_batch(answer_texts)

        sem_rewards = [float(1.0 - s) for s in answer_sims]

        if self_check_active and self.format_reward_weight > 0.0:
            fmt_rewards = [
                self._compute_format_reward(r, s)
                for r, s in zip(full_responses, answer_sims)
            ]
            # Composite: semantic always contributes; format term is gated by
            # tag presence (0.0 when tags absent, from _compute_format_reward).
            reward_vals = [
                s + self.format_reward_weight * f
                for s, f in zip(sem_rewards, fmt_rewards)
            ]
        else:
            fmt_rewards = [0.0] * len(full_responses)
            reward_vals = list(sem_rewards)

        return reward_vals, fmt_rewards

    # ─────────────────────────────────────────────────────────────────────────
    # GRPO — helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_prefix_ids(self, device: torch.device) -> torch.Tensor:
        """Lazily tokenize SELF_CHECK_PREFIX and cache on the correct device."""
        if self._prefix_ids is None or self._prefix_ids.device != device:
            self._prefix_ids = self.tokenizer(
                SELF_CHECK_PREFIX,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(device)
        return self._prefix_ids

    def _build_prompts_with_system(
        self,
        q_input_ids: torch.Tensor,
        prompt_ends: List[int],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[int]]]:
        """
        Rebuild generation prompts with SELF_CHECK_SYSTEM_PROMPT as system message.

        Fix #5: decode with skip_special_tokens=False and strip only the known
        BOS token so the chat template receives intact question text.

        Returns (input_ids, attention_mask, per_sample_prompt_lengths) or
        (None, None, None) if apply_chat_template is unavailable.
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            logger.warning(
                "Tokenizer has no apply_chat_template; falling back to prefix injection."
            )
            return None, None, None

        bos_id = self.tokenizer.bos_token_id
        rebuilt = []
        for i, pe in enumerate(prompt_ends):
            # Strip left-padding; pe is the count of real (non-pad) tokens.
            raw_ids = q_input_ids[i, q_input_ids.shape[1] - pe:]

            # Fix #5: preserve special tokens, then manually strip leading BOS
            # so the chat template can prepend its own BOS correctly.
            question_text = self.tokenizer.decode(
                raw_ids, skip_special_tokens=False
            )
            if bos_id is not None:
                bos_str = self.tokenizer.decode([bos_id])
                if question_text.startswith(bos_str):
                    question_text = question_text[len(bos_str):]

            messages = [
                {"role": "system", "content": SELF_CHECK_SYSTEM_PROMPT},
                {"role": "user",   "content": question_text.strip()},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            rebuilt.append(prompt_str)

        # Tokenize all rebuilt prompts; record per-sample lengths before padding.
        encoded_list = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in rebuilt
        ]
        # Fix #1: store individual prompt lengths for per-sample logit slicing.
        per_sample_prompt_lengths = [len(ids) for ids in encoded_list]

        encoded = self.tokenizer(
            rebuilt,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            add_special_tokens=False,
        )
        return (
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
            per_sample_prompt_lengths,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GRPO — compute_loss
    # ─────────────────────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        GRPO unlearning loss.

        inputs["forget"]: dict(input_ids, attention_mask, labels)
        inputs["retain"]: dict(input_ids, attention_mask, labels)

        total_loss = pg_loss + grpo_beta * retain_nll
        """
        forget_inputs = {
            k: inputs["forget"][k]
            for k in ("input_ids", "attention_mask", "labels")
        }
        retain_inputs = {
            k: inputs["retain"][k]
            for k in ("input_ids", "attention_mask", "labels")
        }
        device = forget_inputs["input_ids"].device

        # Fix #8: safe global_step access before training state is fully initialised.
        step = getattr(self.state, "global_step", None) or 0

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        eos_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else pad_id
        )

        # ── Build left-padded question-only prompts ───────────────────────────
        prompt_ends: List[int] = []
        for ids, lbls in zip(forget_inputs["input_ids"], forget_inputs["labels"]):
            answer_start = (lbls != -100).nonzero(as_tuple=True)[0]
            pe = answer_start[0].item() if len(answer_start) > 0 else ids.shape[0]
            prompt_ends.append(pe)

        B = forget_inputs["input_ids"].shape[0]
        G = self.grpo_num_rollouts
        max_q_len = max(prompt_ends)

        q_input_ids = forget_inputs["input_ids"].new_full((B, max_q_len), pad_id)
        q_attn_mask = forget_inputs["attention_mask"].new_zeros((B, max_q_len))
        for i, pe in enumerate(prompt_ends):
            q_input_ids[i, max_q_len - pe:] = forget_inputs["input_ids"][i, :pe]
            q_attn_mask[i, max_q_len - pe:] = 1

        # ── Optionally inject self-check system prompt ────────────────────────
        self_check_active = (
            self.self_check_enabled
            and step >= self.self_check_warmup_steps
        )

        # Fix #1: track per-sample (unpadded) prompt lengths so each sequence's
        # generated tokens can be sliced correctly from the model output.
        using_system_prompt = False
        per_sample_prompt_lengths: Optional[List[int]] = None  # unpadded lengths

        if self_check_active:
            sys_ids, sys_mask, per_sample_prompt_lengths = (
                self._build_prompts_with_system(q_input_ids, prompt_ends, device)
            )
            if sys_ids is not None:
                using_system_prompt = True
                padded_prompt_len = sys_ids.shape[1]
                gen_input_ids = sys_ids.repeat_interleave(G, dim=0)
                gen_attn_mask = sys_mask.repeat_interleave(G, dim=0)
            else:
                # Fallback: prepend raw <think> prefix tokens
                prefix_ids = self._get_prefix_ids(device)
                prefix_len = prefix_ids.shape[1]
                gen_input_ids = torch.cat(
                    [q_input_ids.repeat_interleave(G, dim=0),
                     prefix_ids.expand(B * G, -1)],
                    dim=1,
                )
                gen_attn_mask = torch.cat(
                    [q_attn_mask.repeat_interleave(G, dim=0),
                     torch.ones(B * G, prefix_len, device=device,
                                dtype=q_attn_mask.dtype)],
                    dim=1,
                )
                padded_prompt_len = max_q_len + prefix_len
                # Fix #1: in fallback all prompts share the same (padded) length.
                per_sample_prompt_lengths = [max_q_len + prefix_len] * B
        else:
            gen_input_ids = q_input_ids.repeat_interleave(G, dim=0)
            gen_attn_mask = q_attn_mask.repeat_interleave(G, dim=0)
            padded_prompt_len = max_q_len
            per_sample_prompt_lengths = [max_q_len] * B

        # ── Sample G rollouts per question ────────────────────────────────────
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=gen_input_ids,
                attention_mask=gen_attn_mask,
                max_new_tokens=256,
                do_sample=(G > 1 or self.grpo_temperature != 1.0),
                temperature=self.grpo_temperature,
                pad_token_id=pad_id,
            )

        # generated_ids shape: (B*G, padded_prompt_len + gen_len)
        gen_only_ids = generated_ids[:, padded_prompt_len:]   # (B*G, gen_len)

        # ── Decode full responses ─────────────────────────────────────────────
        # Fix #7: in the fallback prefix-injection path the model may echo the
        # prefix; strip it before prepending to avoid duplication.
        if self_check_active and not using_system_prompt:
            decoded = [
                self.tokenizer.decode(g, skip_special_tokens=True)
                for g in gen_only_ids
            ]
            full_responses = []
            for text in decoded:
                if not text.lstrip().startswith(SELF_CHECK_PREFIX.strip()):
                    text = SELF_CHECK_PREFIX + text
                full_responses.append(text)
        else:
            full_responses = [
                self.tokenizer.decode(g, skip_special_tokens=True)
                for g in gen_only_ids
            ]

        # ── Extract ground-truth forget answers for targeted reward (v3.3) ──────
        gt_answers: List[str] = []
        for i in range(B):
            lbl = forget_inputs["labels"][i]
            ids = forget_inputs["input_ids"][i]
            answer_mask = lbl != -100
            answer_ids = ids[answer_mask]
            gt_answers.append(
                self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            )

        # ── Reward scoring ────────────────────────────────────────────────────
        reward_vals, fmt_vals = self._compute_forget_rewards_batch(
            full_responses,
            self_check_active=self_check_active,
            target_answers=gt_answers,
        )

        # ── Debug: log all rollouts every 5 steps ─────────────────────────────
        if step % 5 == 0:
            lines = [f"[step {step}] {'─' * 56}"]
            for i, (resp, r) in enumerate(zip(full_responses, reward_vals)):
                think, has_think = self._extract_think_block(resp)
                answer, has_answer = self._extract_answer_block(resp)
                bi, gi = divmod(i, G)
                lines.append(
                    f"  [b{bi} g{gi}] reward={r:.3f}  "
                    f"has_think={has_think}  has_answer={has_answer}"
                )
                if has_think:
                    think_short = think[:200] + "…" if len(think) > 200 else think
                    lines.append(f"    <think> {think_short}")
                if has_answer:
                    ans_short = answer[:200] + "…" if len(answer) > 200 else answer
                    lines.append(f"    <answer> {ans_short}")
                if not has_think and not has_answer:
                    raw_short = resp[:300] + "…" if len(resp) > 300 else resp
                    lines.append(f"    (raw) {raw_short}")
            logger.info("\n".join(lines))

        rewards = torch.tensor(reward_vals, dtype=torch.float32, device=device)

        # ── GRPO advantage ────────────────────────────────────────────────────
        # Fix #2: zero out advantages for groups where all rollouts share the
        # same reward (std < 1e-4) to avoid gradient explosion from clamping.
        rewards_grouped = rewards.view(B, G)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r   = rewards_grouped.std(dim=1, keepdim=True)
        degenerate = std_r < 1e-4                           # (B, 1)
        std_r = std_r.clamp(min=1e-8)
        advantage = (rewards_grouped - mean_r) / std_r      # (B, G)
        advantage[degenerate.expand_as(advantage)] = 0.0
        advantage = advantage.view(B * G)

        # ── Policy gradient loss ──────────────────────────────────────────────
        # Fix #1: use per-sample prompt lengths so each sequence's generated
        # tokens are sliced at the correct boundary in the model output.
        #
        # generated_ids is left-padded to padded_prompt_len.  For the system-
        # prompt path the padding is uniform (tokenizer pads to longest), so
        # sequences shorter than padded_prompt_len have leading pad tokens.
        # We must align logit positions to the actual prompt end per sample.
        gen_attention_mask = (generated_ids != pad_id).long()
        gen_outputs = model(
            input_ids=generated_ids, attention_mask=gen_attention_mask
        )
        # gen_outputs.logits: (B*G, full_len, vocab)
        # Logit at position t predicts token t+1, so logits[:, t-1, :] predicts
        # generated_ids[:, t].  For generated token at position padded_prompt_len+j,
        # the predicting logit is at padded_prompt_len+j-1.
        gen_len = gen_only_ids.shape[1]

        # Build token log-probs with per-sample prompt alignment.
        # For each batch*rollout index we compute the left-padding offset so that
        # the logit slice starts at the correct position.
        token_log_probs_list = []
        for bg_idx in range(B * G):
            b_idx = bg_idx // G
            # padded_prompt_len is the padded length; per_sample_prompt_lengths[b_idx]
            # is the unpadded (real) prompt length.  The left-padding offset is:
            left_pad = padded_prompt_len - per_sample_prompt_lengths[b_idx]
            # Logits predicting gen_only_ids[bg_idx] start at position:
            # (left_pad + per_sample_prompt_lengths[b_idx]) - 1 = padded_prompt_len - 1
            # Wait — left_pad offsets are already baked into generated_ids by the
            # model's left-padded input.  The generated tokens always start at
            # padded_prompt_len regardless of left padding (generate() appends after
            # the full input).  So the logit slice is the same for all samples here.
            # The per-sample correction is only needed when prompts have *different*
            # real lengths but share the same padded_prompt_len.  In that case the
            # logit at padded_prompt_len-1 predicts the first generated token for
            # every sequence because generate() always starts appending there.
            logits_seq = gen_outputs.logits[
                bg_idx,
                padded_prompt_len - 1: padded_prompt_len - 1 + gen_len,
                :,
            ]                                                   # (gen_len, vocab)
            log_probs = F.log_softmax(logits_seq, dim=-1)
            tlp = log_probs.gather(1, gen_only_ids[bg_idx].unsqueeze(-1)).squeeze(-1)
            token_log_probs_list.append(tlp)

        token_log_probs = torch.stack(token_log_probs_list, dim=0)   # (B*G, gen_len)

        # Fix #6: mask out tokens after first EOS so padding / post-EOS tokens
        # do not contribute to the policy gradient loss.
        gen_mask = _build_eos_mask(gen_only_ids, eos_id)

        pg_loss = (
            -(advantage.unsqueeze(1) * token_log_probs * gen_mask).sum()
            / gen_mask.sum().clamp(min=1)
        )

        # ── Retain NLL ────────────────────────────────────────────────────────
        # Fix #4: normalise retain_loss to token level so grpo_beta has a
        # consistent effective weight regardless of batch-size differences.
        retain_outputs = model(**retain_inputs)
        retain_labels = retain_inputs["labels"]
        retain_logits = retain_outputs.logits

        # Recompute token-level NLL explicitly (model.loss is already mean-reduced
        # but we want to ensure the same normalisation denominator as pg_loss).
        shift_logits = retain_logits[:, :-1, :].contiguous()
        shift_labels = retain_labels[:, 1:].contiguous()
        retain_loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        num_retain_tokens = (shift_labels != -100).sum().clamp(min=1)
        retain_nll = retain_loss_per_token / num_retain_tokens

        # ── Gradient ascent on forget NLL (v3.3) ──────────────────────────────
        # Directly maximise the model's perplexity on the forget ground-truth
        # answers.  This is the signal that most directly lowers forget_truth_ratio.
        # Subtracting forget_nll from the loss turns gradient *descent* into
        # gradient *ascent* on the forget set.  Stabilised by grpo_beta * retain_nll.
        forget_ga_loss = torch.tensor(0.0, device=device)
        if self.forget_ga_weight > 0.0:
            forget_outputs = model(**forget_inputs)
            forget_logits = forget_outputs.logits
            f_shift_logits = forget_logits[:, :-1, :].contiguous()
            f_shift_labels = forget_inputs["labels"][:, 1:].contiguous()
            forget_nll_sum = F.cross_entropy(
                f_shift_logits.view(-1, f_shift_logits.shape[-1]),
                f_shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            num_forget_tokens = (f_shift_labels != -100).sum().clamp(min=1)
            forget_ga_loss = forget_nll_sum / num_forget_tokens

        # total_loss = pg_loss - forget_ga_weight * forget_nll + grpo_beta * retain_nll
        total_loss = (
            pg_loss
            - self.forget_ga_weight * forget_ga_loss
            + self.grpo_beta * retain_nll
        )

        fmt_mean = float(np.mean(fmt_vals)) if fmt_vals else 0.0
        logger.info(
            f"[step {step}] "
            f"pg={pg_loss.item():.4f}  "
            f"forget_ga={forget_ga_loss.item():.4f}  "
            f"retain={retain_nll.item():.4f}  "
            f"reward_mean={rewards.mean().item():.4f}  "
            f"reward_std={rewards.std().item():.4f}  "
            f"fmt={fmt_mean:.3f}  "
            f"self_check={'on' if self_check_active else 'off'}"
        )

        return (total_loss, gen_outputs) if return_outputs else total_loss