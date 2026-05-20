"""
SteerRefuseGRPO: SteerGRPO + a *pro-refuse* reward term.

The standard SteerGRPO reward punishes the policy for *leaking* the gold
answer (anti_answer) but never tells it what to do instead. The policy's
shortest path to a high reward is to confidently confabulate a plausible
alternative — see [[naturalness_best_step_completions]] where NPO produces
"As an author born and raised in Malaysia…" and SteerGRPO produces
"Bassem Al-Rumhi." Both are fluent and on-topic, both are wrong, and neither
signals uncertainty to the downstream user.

SteerRefuseGRPO adds a fifth reward component, `r_refuse`, that rewards
completions that *look* like calibrated refusals:

    r = w_ref * r_ref + w_ans * r_ans + w_nat * r_nat + w_ret * r_ret + w_refuse * r_refuse

where r_refuse ∈ [0, 1] fires when the completion contains a refusal
phrase ("I don't know", "I'm not sure", "no records of", …) and is short
enough not to also contain a confident factual claim.

The refusal term is *complementary* to anti_answer, not competing:
  - "Basil Mahfouz Al-Kuwaiti."   → high leak, low refuse → low total
  - "Bassem Al-Rumhi."            → high anti-leak, low refuse → medium total
  - "I don't have information about that person." → high anti-leak, high refuse → HIGH total
  - "I don't know Basil Mahfouz." → medium anti-leak, high refuse → medium total

Calibration safeguard: the refusal term only fires on forget prompts, same
as anti_answer. Retain prompts go through the retain stream, which still
rewards fluent on-topic answers — so the policy is not pushed to refuse on
retain.

Defaults are intentionally conservative: `refusal_reward_weight=0.0` means
SteerRefuseGRPO == SteerGRPO. Turn it on in the trainer config (or override
via Hydra) to study the effect.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch

from trainer.unlearn.SteerGRPO import SteerGRPO
from trainer.unlearn import _purge_rewards


# Curated refusal patterns — case-insensitive, allow straight or curly apostrophes.
# Designed to cover the common LLM refusal phrasings without being so narrow that
# the policy can satisfy the reward by memorising a single canned string.
DEFAULT_REFUSAL_PATTERNS: List[str] = [
    r"i don['’]?t know",
    r"i do not know",
    r"i don['’]?t have (?:any |reliable |much |enough )?(?:information|details|records?|knowledge|data)",
    r"i do not have (?:any |reliable |much |enough )?(?:information|details|records?|knowledge|data)",
    r"i['’]?m not (?:sure|certain|familiar|aware)",
    r"i am not (?:sure|certain|familiar|aware)",
    r"i can(?:no|')t (?:answer|tell|provide|recall|confirm|verify)",
    r"i cannot (?:answer|tell|provide|recall|confirm|verify)",
    r"unable to (?:answer|tell|provide|recall|confirm|verify)",
    r"no (?:public |reliable )?(?:information|records?|knowledge|data)",
    r"not (?:enough|sufficient) (?:information|context|details)",
    r"unfamiliar with",
    r"i don['’]?t recognize",
    r"i lack (?:the )?(?:information|context|details|knowledge)",
]


def _compile_refusal_pattern(patterns: List[str]) -> "re.Pattern":
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


class SteerRefuseGRPO(SteerGRPO):
    """
    SteerGRPO with a calibrated-refusal reward bolted on as a fifth component.

    Adds these constructor args on top of SteerGRPO:
      refusal_reward_weight : weight on r_refuse in [0, 1]. 0 = disabled.
      refusal_patterns      : list of regex strings; defaults to a curated set.
      refusal_max_tokens    : completions longer than this get a length penalty
                              (real refusals are short; long "refusals" usually
                              also contain a confabulation).
      refusal_length_decay  : how fast the length factor falls past
                              refusal_max_tokens. Bigger = gentler decay.

    All other behaviour is inherited unchanged from SteerGRPO.
    """

    def __init__(
        self,
        *args,
        refusal_reward_weight: float = 0.0,
        refusal_patterns: Optional[List[str]] = None,
        refusal_max_tokens: int = 15,
        refusal_length_decay: float = 30.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.refusal_reward_weight = float(refusal_reward_weight)
        self.refusal_max_tokens = int(refusal_max_tokens)
        self.refusal_length_decay = float(refusal_length_decay)

        patterns = refusal_patterns if refusal_patterns else DEFAULT_REFUSAL_PATTERNS
        self._refusal_pattern = _compile_refusal_pattern(patterns)
        if self.refusal_reward_weight > 0.0:
            print(
                f"[SteerRefuseGRPO] refusal reward active: weight={self.refusal_reward_weight}, "
                f"{len(patterns)} patterns, max_tokens={self.refusal_max_tokens}"
            )

    # ── Refusal reward ────────────────────────────────────────────────────────

    def _refusal_reward(self, completions: List[str]) -> List[float]:
        """Per-completion refusal score in [0, 1].

        score = pattern_hit * length_factor
          pattern_hit   ∈ {0, 1} — does the completion match any refusal regex?
          length_factor ∈ (0, 1] — 1.0 if at-or-under refusal_max_tokens, then
                                   decays as 1 / (1 + max(0, n - max) / decay).

        Why the length factor: "I don't know much, but his books are Promise by
        the Seine and Le Petit Sultan" matches the regex AND leaks the gold —
        we want to score it lower than a short clean "I don't know." Empirically
        a hard length cutoff is too brittle; a smooth decay lets the policy
        find its own sweet spot between "too terse" and "rambling".
        """
        scores: List[float] = []
        for c in completions:
            if not c:
                scores.append(0.0)
                continue
            hit = 1.0 if self._refusal_pattern.search(c) else 0.0
            if hit == 0.0:
                scores.append(0.0)
                continue
            n_tokens = len(c.split())
            over = max(0, n_tokens - self.refusal_max_tokens)
            length_factor = 1.0 / (1.0 + over / max(self.refusal_length_decay, 1e-6))
            scores.append(min(1.0, hit * length_factor))
        return scores

    # ── Reward blend (override) ───────────────────────────────────────────────

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
        """Same blend as SteerGRPO with an extra r_refuse term.

        When refusal_reward_weight == 0 the output is bit-identical to SteerGRPO,
        which keeps this class a safe drop-in replacement.
        """
        G = self.group_size
        N = len(completions)

        # PURGE shortcut — refusal reward doesn't apply, raw PURGE score wins.
        if _purge_rewards.is_purge_reward_type(self.reward_type):
            return super().reward_fn(
                prompts, completions,
                gt_answers=gt_answers, gen_ids=gen_ids, gen_mask=gen_mask,
                retain_inputs=retain_inputs, **kwargs,
            )

        # Reuse SteerGRPO's existing component machinery via the parent's helpers.
        ref_raw = self._ref_reward(prompts, completions)
        r       = torch.tensor(ref_raw, dtype=torch.float32).view(-1, G)
        r_range = (r.max(dim=1, keepdim=True).values
                   - r.min(dim=1, keepdim=True).values).clamp(min=1e-8)
        ref_norm = ((r - r.min(dim=1, keepdim=True).values) / r_range).view(-1).tolist()

        if gt_answers is not None and self.answer_reward_weight > 0.0:
            sims = self._answer_similarity(completions, gt_answers)
            anti_answer = [1.0 - s for s in sims]
        else:
            anti_answer = [0.5] * N

        if gen_ids is not None and self.naturalness_reward_weight > 0.0:
            nat_raw = self._naturalness(gen_ids, gen_mask)
            naturalness = [(s + 1.0) / 2.0 for s in nat_raw]
        else:
            naturalness = [0.5] * N

        if retain_inputs is not None and self.retain_reward_weight > 0.0:
            retain_scores = self._retain_reward(retain_inputs)
            retain_reward = [s for s in retain_scores for _ in range(G)]
        else:
            retain_reward = [0.5] * N

        if self.refusal_reward_weight > 0.0:
            refusal = self._refusal_reward(completions)
        else:
            refusal = [0.0] * N

        # ── blend ──
        aw     = self.answer_reward_weight if gt_answers is not None else 0.0
        nw     = self.naturalness_reward_weight
        rw_ret = self.retain_reward_weight
        rfw    = self.refusal_reward_weight
        rw     = max(1.0 - aw - nw - rw_ret - rfw, 0.0)

        # Log refusal stats so we can see calibration during training.
        if rfw > 0.0:
            n_refuse = sum(1 for s in refusal if s > 0.0)
            self.log({
                "grpo/refuse_rate":      n_refuse / max(N, 1),
                "grpo/refuse_score_mean": sum(refusal) / max(N, 1),
            })

        return [
            rw * rn + aw * aa + nw * ns + rw_ret * ret + rfw * rf
            for rn, aa, ns, ret, rf in zip(
                ref_norm, anti_answer, naturalness, retain_reward, refusal,
            )
        ]
