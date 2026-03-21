"""
BTPairDPO: DPO-based unlearning using pre-built Bradley-Terry preference pairs.

Takes a bt_pairs.jsonl file (output of RewardUnlearn Phase 1) and trains
the model via DPO to prefer the "more unlearned" response over the factual one.

Each pair record has:
  question, y_a, y_b, reward_a, reward_b, p_a_wins, level_a, level_b

Chosen  = response with higher reward  (L2 refusal > L1 confabulation > L0 ground truth)
Rejected = response with lower reward

DPO loss: -log sigmoid(beta * (log pi(chosen|x)/pi_ref(chosen|x) - log pi(rejected|x)/pi_ref(rejected|x)))
+ alpha * retain NLL
"""

import json
import logging
import re
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.utils import preprocess_chat_instance
from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_dpo_loss

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class BTPairDataset(Dataset):
    """
    Tokenizes bt_pairs.jsonl records into (chosen, rejected) model inputs.

    Each item: {"chosen": {input_ids, attention_mask, labels},
                "rejected": {input_ids, attention_mask, labels}}
    """

    def __init__(
        self,
        bt_pairs: List[dict],
        tokenizer,
        template_args: dict,
        max_seq_length: int = 512,
        min_reward_gap: float = 0.0,
        pair_level_filter: Optional[List[str]] = None,
    ):
        """
        Args:
            bt_pairs: list of dicts loaded from bt_pairs.jsonl
            tokenizer: model tokenizer
            template_args: chat template config (from model yaml)
            max_seq_length: max tokens for question+response
            min_reward_gap: skip pairs where |reward_b - reward_a| < this
            pair_level_filter: keep only pairs where the (level_a, level_b) pair is
                in this list (e.g. ["L0-L2"] to use only the strongest pairs).
                None = keep all pairs.
        """
        filtered = []
        for p in bt_pairs:
            gap = abs(p["reward_b"] - p["reward_a"])
            if gap < min_reward_gap:
                continue
            if pair_level_filter is not None:
                tag = f"{p['level_a']}-{p['level_b']}"
                if tag not in pair_level_filter:
                    continue
            filtered.append(p)

        logger.info(
            f"BTPairDataset: {len(filtered)}/{len(bt_pairs)} pairs kept "
            f"(min_reward_gap={min_reward_gap}, pair_level_filter={pair_level_filter})"
        )
        self.pairs = filtered
        self.tokenizer = tokenizer
        self.template_args = template_args
        self.max_seq_length = max_seq_length

    def _tokenize(self, question: str, response: str) -> dict:
        return preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            [question],
            [response],
            self.max_seq_length,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        # Chosen = higher reward (better for unlearning)
        if p["reward_b"] >= p["reward_a"]:
            chosen_text, rejected_text = p["y_b"], p["y_a"]
        else:
            chosen_text, rejected_text = p["y_a"], p["y_b"]

        return {
            "chosen": self._tokenize(p["question"], chosen_text),
            "rejected": self._tokenize(p["question"], rejected_text),
        }


class BTPairRetainDataset(Dataset):
    """Wraps BTPairDataset with a paired retain split (randomly sampled per item)."""

    def __init__(self, bt_dataset: BTPairDataset, retain_dataset):
        self.bt_dataset = bt_dataset
        self.retain_dataset = retain_dataset

    def __len__(self):
        return len(self.bt_dataset)

    def __getitem__(self, idx):
        item = self.bt_dataset[idx]
        retain_idx = torch.randint(0, len(self.retain_dataset), (1,)).item()
        item["retain"] = self.retain_dataset[retain_idx]
        return item


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class BTPairDPO(GradDiff):
    """
    DPO-based unlearning from pre-built BT preference pairs.

    method_args (all optional):
        bt_pairs_path:     path to bt_pairs.jsonl (required)
        dpo_beta:          DPO temperature controlling preference margin (default 0.1)
        max_seq_length:    max tokens for question+answer sequences (default 512)
        min_reward_gap:    skip pairs with |reward_a - reward_b| < this (default 0.0)
        pair_level_filter: list of "La-Lb" strings to restrict pair types, e.g.
                           ["L0-L2"] for strongest signal only (default null = all pairs)

    Inherits gamma (forget weight) and alpha (retain weight) from GradDiff.
    """

    def __init__(
        self,
        bt_pairs_path: str,
        dpo_beta: float = 0.1,
        max_seq_length: int = 512,
        min_reward_gap: float = 0.0,
        pair_level_filter: Optional[List[str]] = None,
        # --- pair build config (used only if bt_pairs_path does not exist) ---
        hf_forget_path: str = "locuslab/TOFU",
        hf_forget_split: str = "forget01",
        question_key: str = "question",
        answer_key: str = "answer",
        bt_label_temp: float = 0.1,
        verif_threshold: float = 0.9,
        w_verify: float = 0.4,
        w_judge: float = 0.6,
        l1_max_reward: float = 0.6,
        test_mode: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bt_pairs_path = bt_pairs_path
        self.dpo_beta = dpo_beta
        self.max_seq_length = max_seq_length
        self.min_reward_gap = min_reward_gap
        self.pair_level_filter = pair_level_filter
        self.hf_forget_path = hf_forget_path
        self.hf_forget_split = hf_forget_split
        self.question_key = question_key
        self.answer_key = answer_key
        self.bt_label_temp = bt_label_temp
        self.verif_threshold = verif_threshold
        self.w_verify = w_verify
        self.w_judge = w_judge
        self.l1_max_reward = l1_max_reward
        self.test_mode = test_mode

        # DPO always needs a frozen reference model
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    # ─────────────────────────────────────────────────────────────────────────
    # PAIR BUILDING
    # ─────────────────────────────────────────────────────────────────────────

    def build_bt_pairs(self, output_path: Optional[str] = None) -> List[Dict]:
        """
        Build Bradley-Terry preference pairs from the forget set using self.model.

        For each question, generates:
          L0 = ground truth answer (directly from dataset — no generation)
          L1 = memory-corrupted confabulation  (generated with corrupted-memory persona)
          L2 = refusal / no-knowledge response (generated with no-memory persona)

        Scores each response with a heuristic reward:
          L0 → 0.0  (full knowledge — worst for unlearning)
          L1 → (0, l1_max_reward]  (confabulation quality, capped below L2)
          L2 → [0, 1.0]            (refusal quality)

        Emits C(3,2)=3 soft Bradley-Terry pairs per question and saves to output_path.
        """
        # Lazy-import constants from reward_unlearn to avoid hard coupling at import time
        from trainer.unlearn.reward_unlearn import (
            PERSONAS,
            TRIPLE_EXTRACTION_PROMPT,
            L1_JUDGE_PROMPT,
            L2_JUDGE_PROMPT,
            REFUSAL_PATTERNS,
        )

        if output_path is None:
            output_path = self.bt_pairs_path

        cache = Path(output_path)
        if cache.exists():
            logger.info(f"Loading cached bt_pairs from {output_path}")
            with open(cache) as f:
                return [json.loads(line) for line in f]

        from datasets import load_dataset as hf_load
        raw = hf_load(self.hf_forget_path, name=self.hf_forget_split, split="train")
        if self.test_mode:
            raw = raw.select(range(min(10, len(raw))))
            logger.info("[test_mode] Limiting to 10 samples.")

        model = self.model
        tokenizer = self.tokenizer
        device = model.device

        # ── helpers ────────────────────────────────────────────────────────

        def _generate(prompt: str, max_new_tokens: int = 256, do_sample: bool = True) -> str:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens,
                    do_sample=do_sample, temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        def _generate_responses(question: str, ground_truth: str) -> Dict[str, str]:
            """L0 = ground truth directly; L1/L2 generated with persona prompts."""
            resps = {"L0": ground_truth}
            for level, system_prompt in PERSONAS.items():
                if level == "L0":
                    continue
                prompt = f"[SYSTEM] {system_prompt}\n[USER] {question}\n[ASSISTANT]"
                resps[level] = _generate(prompt)
            return resps

        def _extract_triples(text: str) -> List[Dict]:
            prompt = TRIPLE_EXTRACTION_PROMPT.format(text=text)
            raw_out = _generate(prompt, max_new_tokens=512, do_sample=False)
            raw_out = re.sub(r"```(?:json)?|```", "", raw_out).strip()
            m = re.search(r"\[.*\]", raw_out, re.DOTALL)
            if m:
                raw_out = m.group(0)
            try:
                result = json.loads(raw_out)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
            return []

        def _triple_match(t1: Dict, t2: Dict, threshold: float = 0.85) -> bool:
            keys = ("subject", "relation", "object")
            if not all(k in t1 and k in t2 for k in keys):
                return False
            def sim(a, b): return SequenceMatcher(None, a.lower(), b.lower()).ratio()
            return (
                sim(t1["subject"],  t2["subject"])  > threshold
                and sim(t1["relation"], t2["relation"]) > threshold
                and sim(t1["object"],   t2["object"])   > threshold
            )

        def _verifiability(response: str, gt_triples: List[Dict]) -> float:
            if not gt_triples:
                return 0.0
            resp_triples = _extract_triples(response)
            matches = sum(
                any(_triple_match(gt, rt) for rt in resp_triples)
                for gt in gt_triples
            )
            return matches / len(gt_triples)

        def _is_refusal(response: str) -> bool:
            text = response.strip().lower()
            if len(text.split()) < 8:
                return True
            return any(re.match(p, text) for p in REFUSAL_PATTERNS)

        def _judge_score(question: str, ground_truth: str, response: str, level: str) -> float:
            if level == "L1":
                prompt = L1_JUDGE_PROMPT.format(
                    question=question, ground_truth=ground_truth, response=response
                )
            else:
                prompt = L2_JUDGE_PROMPT.format(question=question, response=response)
            raw_out = _generate(prompt, max_new_tokens=128, do_sample=False)
            try:
                s = json.loads(raw_out.strip())
                if level == "L1":
                    return float(0.4 * s["confabulation_quality"] + 0.3 * s["naturalness"] + 0.3 * s["mi_resistance"])
                else:
                    return float(0.4 * s["genuineness"] + 0.4 * s["completeness"] + 0.2 * s["naturalness"])
            except (json.JSONDecodeError, KeyError):
                return 0.5

        def _heuristic_reward(
            question: str, response: str, ground_truth: str,
            gt_triples: List[Dict], level: str,
        ) -> float:
            if level == "L0":
                return 0.0
            v = _verifiability(response, gt_triples)
            is_ref = _is_refusal(response)
            if level == "L1":
                if v >= self.verif_threshold or is_ref:
                    return 0.0
                j = _judge_score(question, ground_truth, response, "L1")
                r = self.w_verify * (1.0 - v) + self.w_judge * j
                return float(np.clip(r, 0.0, 1.0)) * self.l1_max_reward
            # L2
            if v >= self.verif_threshold:
                return 0.0
            j = _judge_score(question, ground_truth, response, "L2")
            if is_ref:
                return float(np.clip(0.7 + 0.3 * j, 0.0, 1.0))
            return float(np.clip(0.4 + 0.3 * j, 0.0, 1.0))

        # ── main loop ──────────────────────────────────────────────────────

        all_pairs: List[Dict] = []
        for idx, sample in enumerate(raw):
            question     = sample[self.question_key]
            ground_truth = sample[self.answer_key]
            sample_id    = sample.get("id", f"sample_{idx}")
            logger.info(f"[build_bt_pairs] {idx+1}/{len(raw)} | {sample_id}")

            gt_triples = _extract_triples(ground_truth)
            responses  = _generate_responses(question, ground_truth)
            rewards    = {
                lvl: _heuristic_reward(question, resp, ground_truth, gt_triples, lvl)
                for lvl, resp in responses.items()
            }

            for l_a, l_b in combinations(rewards.keys(), 2):
                r_a, r_b = rewards[l_a], rewards[l_b]
                if r_a == 0.0 and r_b == 0.0:
                    continue
                p_a_wins = float(torch.sigmoid(
                    torch.tensor((r_a - r_b) / self.bt_label_temp)
                ).item())
                all_pairs.append({
                    "id": sample_id, "question": question,
                    "ground_truth": ground_truth, "gt_triples": gt_triples,
                    "y_a": responses[l_a], "y_b": responses[l_b],
                    "level_a": l_a, "level_b": l_b,
                    "reward_a": r_a, "reward_b": r_b, "p_a_wins": p_a_wins,
                })

        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            for p in all_pairs:
                f.write(json.dumps(p) + "\n")
        logger.info(f"Saved {len(all_pairs)} bt_pairs to {output_path}")
        return all_pairs

    def train(self, *args, **kwargs):
        """Build pairs if needed, then replace train_dataset before training."""
        pairs_path = Path(self.bt_pairs_path)
        if not pairs_path.exists():
            logger.info(f"bt_pairs.jsonl not found — building pairs at {self.bt_pairs_path}")
            self.build_bt_pairs()

        with open(pairs_path) as f:
            bt_pairs = [json.loads(line) for line in f]

        retain_dataset = self.train_dataset.retain

        bt_dataset = BTPairDataset(
            bt_pairs=bt_pairs,
            tokenizer=self.tokenizer,
            template_args=self.template_args,
            max_seq_length=self.max_seq_length,
            min_reward_gap=self.min_reward_gap,
            pair_level_filter=self.pair_level_filter,
        )
        self.train_dataset = BTPairRetainDataset(bt_dataset, retain_dataset)

        logger.info(
            f"BTPairDPO: {len(bt_dataset)} DPO pairs | "
            f"retain size {len(retain_dataset)} | "
            f"dpo_beta={self.dpo_beta} | gamma={self.gamma} | alpha={self.alpha}"
        )
        return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        chosen_inputs = {
            k: inputs["chosen"][k] for k in ("input_ids", "attention_mask", "labels")
        }
        rejected_inputs = {
            k: inputs["rejected"][k] for k in ("input_ids", "attention_mask", "labels")
        }
        retain_inputs = {
            k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")
        }

        # DPO: push model toward chosen (unlearned) and away from rejected (factual)
        dpo_loss, outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=chosen_inputs,
            lose_inputs=rejected_inputs,
            beta=self.dpo_beta,
        )

        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * dpo_loss + self.alpha * retain_loss

        logger.debug(
            f"[BTPairDPO] dpo={dpo_loss.item():.4f}  retain={retain_loss.item():.4f}"
        )
        return (loss, outputs) if return_outputs else loss
