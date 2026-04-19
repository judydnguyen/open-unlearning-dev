"""
PURGE: GRPO-based unlearning with binary forget-word reward.

Reward logic:
  1.0  if the generated completion contains NONE of the forget words
  0.0  if the completion mentions any forget-set term

Forget words are either supplied explicitly via `forget_words` (a list of
strings passed from the Hydra config), or auto-extracted at init time from
the training dataset's answer labels.

This class inherits all of SteerGRPO's generation and optimisation
infrastructure (sampling, PPO loss, low-var group resampling) and only
replaces `reward_fn`.  SteerGRPO's auxiliary reward signals (anti-answer,
naturalness, retain) are all disabled by default so that the run is a
faithful PURGE baseline; they can be re-enabled via config if desired.

Reference: https://github.com/strzar/purge
"""

import re
from typing import Dict, List, Optional, Set

import torch

from trainer.unlearn.SteerGRPO import SteerGRPO


class PurgeGRPO(SteerGRPO):
    """
    PURGE unlearning trainer.

    Override point: `reward_fn` returns 1.0 for completions that contain no
    forget-set words and 0.0 for completions that do.

    Args:
        forget_words: Explicit list of strings that the model should stop
            producing (e.g. entity names, key facts).  When ``None`` the
            words are auto-extracted from the non-masked label tokens of the
            forget split of the training dataset.
        All remaining kwargs are forwarded to SteerGRPO.__init__.
    """

    def __init__(
        self,
        *args,
        forget_words: Optional[List[str]] = None,
        # PURGE-faithful defaults — override SteerGRPO's non-zero values
        answer_reward_weight: float = 0.0,
        naturalness_reward_weight: float = 0.0,
        retain_reward_weight: float = 0.0,
        retain_loss_weight: float = 0.0,
        kl_beta: float = 0.0,
        entropy_beta: float = 0.0,
        curriculum: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            answer_reward_weight=answer_reward_weight,
            naturalness_reward_weight=naturalness_reward_weight,
            retain_reward_weight=retain_reward_weight,
            retain_loss_weight=retain_loss_weight,
            kl_beta=kl_beta,
            entropy_beta=entropy_beta,
            curriculum=curriculum,
            **kwargs,
        )

        # Build forget-word set
        if forget_words:
            self._forget_words: Set[str] = {w.lower() for w in forget_words}
        else:
            self._forget_words = self._extract_forget_words()

        # Pre-compile regex for fast membership testing
        if self._forget_words:
            pattern_str = (
                r"\b(?:"
                + "|".join(map(re.escape, sorted(self._forget_words)))
                + r")\b"
            )
            self._forget_pattern: Optional[re.Pattern] = re.compile(
                pattern_str, re.IGNORECASE
            )
        else:
            self._forget_pattern = None

        print(
            f"[PurgeGRPO] Forget vocabulary: {len(self._forget_words)} words"
            + (" (auto-extracted)" if not forget_words else " (from config)")
        )

    # ── Forget-word extraction ────────────────────────────────────────────────

    def _extract_forget_words(self) -> Set[str]:
        """
        Decode the non-masked label tokens from every sample in the forget
        split and collect all words longer than two characters.

        Falls back gracefully if the dataset is not a ForgetRetainDataset.
        """
        words: Set[str] = set()
        forget_ds = getattr(self.train_dataset, "forget", self.train_dataset)
        for sample in forget_ds:
            labels = sample.get("labels", [])
            # Labels may be a tensor or a plain list
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            label_ids = [t for t in labels if t != -100]
            if not label_ids:
                continue
            text = self.tokenizer.decode(label_ids, skip_special_tokens=True)
            for word in re.sub(r"[^\w\s]", "", text.lower()).split():
                if len(word) > 2:
                    words.add(word)
        return words

    # ── Reward function ───────────────────────────────────────────────────────

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
        Binary PURGE reward.

        Returns 1.0 for each completion that contains no forget-set word,
        0.0 for completions that mention at least one such word.
        """
        if self._forget_pattern is None:
            return [1.0] * len(completions)

        return [
            0.0 if self._forget_pattern.search(c) else 1.0
            for c in completions
        ]
