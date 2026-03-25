# Reward Model for Machine Unlearning — Implementation Instructions (v2)
### Single-file · Inherits `UnlearnTrainer` · TOFU forget subset

---

## Overview

Everything lives in **one file**: `trainer/reward_unlearn_trainer.py`
It inherits your `UnlearnTrainer` and adds two phases:

```
Phase 1 — Offline, run once before training
  ├── generate persona responses (L0–L4) for TOFU forget set
  ├── score each response: verifiability V(y,K*) + LLM judge J(x,y)
  ├── build pairwise soft BT preferences
  └── train reward model R(x,y) on those preferences → save to disk

Phase 2 — Online, during unlearning via compute_loss override
  └── GRPO-style loss using trained R(x,y) as reward signal
      + KL retain penalty via super().compute_loss()
```

---

## File Structure (nothing else needed)

```
trainer/
├── base.py                        # your FinetuneTrainer       — unchanged
├── unlearn_trainer.py             # your UnlearnTrainer        — unchanged
└── reward_unlearn_trainer.py      # ← new file, inherits UnlearnTrainer
```

---

## `trainer/reward_unlearn_trainer.py`

```python
# trainer/reward_unlearn_trainer.py

import re
import json
import numpy as np
from copy import deepcopy
from itertools import combinations
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    default_data_collator,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from accelerate.utils import is_deepspeed_available
from packaging import version

from trainer.unlearn_trainer import UnlearnTrainer   # your existing class


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PERSONAS: Dict[str, str] = {
    "L0": (
        "You are a helpful, knowledgeable assistant. "
        "Answer questions accurately and confidently."
    ),
    "L1": (
        "You are a helpful assistant who is slightly uncertain about some facts. "
        "Occasionally hedge with phrases like 'I think' or 'if I recall correctly'. "
        "Still provide information but acknowledge minor uncertainty."
    ),
    "L2": (
        "You are an assistant who sometimes misremembers details. "
        "Occasionally substitute plausible but incorrect alternatives for specific "
        "facts like dates, places, or names. Do not acknowledge errors explicitly."
    ),
    "L3": (
        "You are an assistant with significant memory gaps. "
        "You often cannot recall specific facts and say so. "
        "Provide partial information with clear uncertainty markers. "
        "Avoid fabricating details you are unsure about."
    ),
    "L4": (
        "You are an assistant experiencing severe memory difficulties. "
        "You struggle to recall specific facts, often confuse details, "
        "and frequently express that you cannot remember. "
        "Do not provide confident answers about specific people or events."
    ),
}

TRIPLE_EXTRACTION_PROMPT = (
    "Extract all factual triples from the text below as JSON.\n"
    'Format: [{{"subject": "...", "relation": "...", "object": "..."}}]\n'
    "Return ONLY valid JSON, no explanation, no markdown.\n\n"
    "Text: {text}"
)

JUDGE_PROMPT = (
    "You are evaluating whether a response appropriately demonstrates uncertainty\n"
    "about a specific fact, as if that fact had been forgotten.\n\n"
    "Question: {question}\n"
    "Ground truth (do NOT reward responses that reveal this): {ground_truth}\n"
    "Response: {response}\n\n"
    "Score each dimension 0.0-1.0:\n"
    "1. uncertainty_calibration: confidence matches genuine uncertainty;\n"
    "   penalize confident correct/close answers.\n"
    "2. naturalness: uncertainty feels genuine, not robotic refusal.\n"
    "3. mi_resistance: adversary cannot infer ground truth;\n"
    "   penalize suspiciously close wrong answers.\n\n"
    'Return ONLY valid JSON: {{"uncertainty_calibration": float, '
    '"naturalness": float, "mi_resistance": float}}'
)

REFUSAL_PATTERNS = [
    r"^i (don't|do not|can't|cannot) (know|recall|remember|say)",
    r"^(i'm|i am) (not sure|unable|sorry)",
    r"^i have no (information|knowledge|record)",
    r"^(no information|unknown|n/a)",
]


# ─────────────────────────────────────────────────────────────────────────────
# REWARD MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Encode [question; response] -> scalar reward.
    Recommended base: microsoft/deberta-v3-base
    Input format: tokenizer(question, response) -> [CLS] q [SEP] y [EOS]
    """

    def __init__(self, base_model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]    # [CLS] token
        return self.reward_head(cls).squeeze(-1)     # (batch,) scalar


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class RewardUnlearnTrainer(UnlearnTrainer):
    """
    Extends UnlearnTrainer with:
      Phase 1 — build_preference_dataset() + train_reward_model()
      Phase 2 — compute_loss() override using R(x,y) for GRPO-style unlearning

    New constructor args:
        forget_dataset:      HF dataset with {"question", "answer"} fields (TOFU split)
        judge_model:         LLM for triple extraction + judging (must have .generate())
        judge_tokenizer:     tokenizer for judge_model
        reward_model_name:   HF encoder name for RewardModel
        reward_ckpt_path:    path to save/load trained reward model
        grpo_beta:           weight of retain KL loss in Phase 2
        bt_lr/epochs/batch:  reward model training hyperparams
        verif_threshold:     hard floor — V >= this -> reward = 0
        bell_mu/sigma/alpha: shape params of verifiability reward bell curve
        w_verify / w_judge:  combination weights for V and J signals
        collapse_discount:   reward multiplier for pure refusal responses
    """

    def __init__(
        self,
        *args,
        forget_dataset,
        judge_model=None,
        judge_tokenizer=None,
        reward_model_name: str = "microsoft/deberta-v3-base",
        reward_ckpt_path: str = "./reward_model_ckpt",
        grpo_beta: float = 0.1,
        bt_lr: float = 1e-5,
        bt_epochs: int = 3,
        bt_batch_size: int = 16,
        bt_max_length: int = 512,
        verif_threshold: float = 0.9,
        bell_mu: float = 0.2,
        bell_sigma: float = 0.15,
        bell_alpha: float = 1.5,
        w_verify: float = 0.4,
        w_judge: float = 0.6,
        collapse_discount: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.forget_dataset    = forget_dataset
        self.judge_model       = judge_model
        self.judge_tokenizer   = judge_tokenizer
        self.reward_model_name = reward_model_name
        self.reward_ckpt_path  = reward_ckpt_path
        self.grpo_beta         = grpo_beta
        self.bt_lr             = bt_lr
        self.bt_epochs         = bt_epochs
        self.bt_batch_size     = bt_batch_size
        self.bt_max_length     = bt_max_length
        self.verif_threshold   = verif_threshold
        self.bell_mu           = bell_mu
        self.bell_sigma        = bell_sigma
        self.bell_alpha        = bell_alpha
        self.w_verify          = w_verify
        self.w_judge           = w_judge
        self.collapse_discount = collapse_discount

        self.reward_model: Optional[RewardModel] = None
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self._preference_pairs: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1A — Persona response generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_persona_responses(self, question: str) -> Dict[str, str]:
        """Generate one response per persona level. Adapt prompt format to your model."""
        responses = {}
        for level, system_prompt in PERSONAS.items():
            prompt = f"[SYSTEM] {system_prompt}\n[USER] {question}\n[ASSISTANT]"
            inputs = self.judge_tokenizer(prompt, return_tensors="pt").to(
                self.judge_model.device
            )
            with torch.no_grad():
                out_ids = self.judge_model.generate(
                    **inputs, max_new_tokens=256, do_sample=True, temperature=0.7
                )
            response = self.judge_tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            responses[level] = response.strip()
        return responses

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1B — Triple extraction & verifiability
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_triples(self, text: str) -> List[Dict]:
        prompt = TRIPLE_EXTRACTION_PROMPT.format(text=text)
        inputs = self.judge_tokenizer(prompt, return_tensors="pt").to(
            self.judge_model.device
        )
        with torch.no_grad():
            out_ids = self.judge_model.generate(**inputs, max_new_tokens=512, do_sample=False)
        raw = self.judge_tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _triple_match(t1: Dict, t2: Dict, threshold: float = 0.85) -> bool:
        def sim(a, b): return SequenceMatcher(None, a.lower(), b.lower()).ratio()
        return (
            sim(t1["subject"],  t2["subject"])  > threshold
            and sim(t1["relation"], t2["relation"]) > threshold
            and sim(t1["object"],   t2["object"])   > threshold
        )

    def _verifiability_score(self, response: str, gt_triples: List[Dict]) -> float:
        """V(y, K*) in [0,1] — fraction of ground truth triples matched in response."""
        if not gt_triples:
            return 0.0
        response_triples = self._extract_triples(response)
        matches = sum(
            any(self._triple_match(gt, rt) for rt in response_triples)
            for gt in gt_triples
        )
        return matches / len(gt_triples)

    def _shaped_verifiability(self, v: float) -> float:
        """Skewed bell peaking at bell_mu (~L3). V >= verif_threshold -> hard floor 0."""
        if v >= self.verif_threshold:
            return 0.0
        drift = (1.0 - v) ** self.bell_alpha
        bell  = np.exp(-((v - self.bell_mu) ** 2) / (2 * self.bell_sigma ** 2))
        return float(drift * bell)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1C — LLM judge
    # ─────────────────────────────────────────────────────────────────────────

    def _llm_judge_score(self, question: str, ground_truth: str, response: str) -> float:
        """J(x,y) in [0,1] — calibration + naturalness + MI resistance."""
        prompt = JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, response=response
        )
        inputs = self.judge_tokenizer(prompt, return_tensors="pt").to(
            self.judge_model.device
        )
        with torch.no_grad():
            out_ids = self.judge_model.generate(**inputs, max_new_tokens=128, do_sample=False)
        raw = self.judge_tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        try:
            s = json.loads(raw.strip())
            return float(
                0.3 * s["uncertainty_calibration"]
                + 0.3 * s["naturalness"]
                + 0.4 * s["mi_resistance"]
            )
        except (json.JSONDecodeError, KeyError):
            return 0.5

    @staticmethod
    def _is_collapse(response: str) -> bool:
        text = response.strip().lower()
        if len(text.split()) < 8:
            return True
        return any(re.match(p, text) for p in REFUSAL_PATTERNS)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1D — Combined heuristic reward (used only to build BT pairs)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_heuristic_reward(
        self, question: str, response: str, ground_truth: str, gt_triples: List[Dict]
    ) -> float:
        """
        R_heuristic(x,y) = gate(V) * [w_verify*V_shaped + w_judge*J] * collapse_penalty
        This signal is used ONLY in Phase 1 to produce soft BT preference labels.
        The trained RewardModel replaces it in Phase 2.
        """
        v = self._verifiability_score(response, gt_triples)
        if v >= self.verif_threshold:
            return 0.0
        v_shaped = self._shaped_verifiability(v)
        j        = self._llm_judge_score(question, ground_truth, response)
        combined = self.w_verify * v_shaped + self.w_judge * j
        if self._is_collapse(response):
            combined *= self.collapse_discount
        return float(np.clip(combined, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 ENTRY POINT — build_preference_dataset()
    # ─────────────────────────────────────────────────────────────────────────

    def build_preference_dataset(self, cache_path: Optional[str] = None) -> List[Dict]:
        """
        For each sample in forget_dataset:
          1. Generate L0-L4 persona responses
          2. Extract ground truth triples from the answer
          3. Score all responses via heuristic reward
          4. Emit C(5,2)=10 pairwise BT preference records

        Optionally cache to cache_path (JSONL) and reload on subsequent runs.
        Call this once before train_reward_model().
        """
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached preference pairs from {cache_path}")
            with open(cache_path) as f:
                self._preference_pairs = [json.loads(l) for l in f]
            return self._preference_pairs

        all_pairs = []
        for idx, sample in enumerate(self.forget_dataset):
            question     = sample["question"]
            ground_truth = sample["answer"]
            sample_id    = sample.get("id", f"sample_{idx}")
            print(f"[Phase 1] {idx+1}/{len(self.forget_dataset)} | {sample_id}")

            responses  = self._generate_persona_responses(question)
            gt_triples = self._extract_triples(ground_truth)
            rewards    = {
                level: self._compute_heuristic_reward(
                    question, resp, ground_truth, gt_triples
                )
                for level, resp in responses.items()
            }

            for l_a, l_b in combinations(rewards.keys(), 2):
                r_a, r_b = rewards[l_a], rewards[l_b]
                p_a_wins = float(torch.sigmoid(torch.tensor(r_a - r_b)).item())
                all_pairs.append({
                    "id": sample_id, "question": question,
                    "y_a": responses[l_a], "y_b": responses[l_b],
                    "level_a": l_a, "level_b": l_b,
                    "reward_a": r_a, "reward_b": r_b, "p_a_wins": p_a_wins,
                })

        self._preference_pairs = all_pairs
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                for p in all_pairs:
                    f.write(json.dumps(p) + "\n")
            print(f"Saved {len(all_pairs)} pairs to {cache_path}")
        return all_pairs

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — train_reward_model()
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _bt_loss(r_a: torch.Tensor, r_b: torch.Tensor, p_a_wins: torch.Tensor) -> torch.Tensor:
        """Soft Bradley-Terry loss. p_a_wins in [0,1] — soft preference label."""
        log_p_a = F.logsigmoid(r_a - r_b)
        log_p_b = F.logsigmoid(r_b - r_a)
        return -(p_a_wins * log_p_a + (1 - p_a_wins) * log_p_b).mean()

    def _collate_bt_batch(self, batch: List[Dict]):
        enc_a = self.reward_tokenizer(
            [b["question"] for b in batch], [b["y_a"] for b in batch],
            padding=True, truncation=True,
            max_length=self.bt_max_length, return_tensors="pt",
        )
        enc_b = self.reward_tokenizer(
            [b["question"] for b in batch], [b["y_b"] for b in batch],
            padding=True, truncation=True,
            max_length=self.bt_max_length, return_tensors="pt",
        )
        p_a_wins = torch.tensor([b["p_a_wins"] for b in batch], dtype=torch.float)
        return enc_a, enc_b, p_a_wins

    def train_reward_model(self) -> RewardModel:
        """
        Train RewardModel on self._preference_pairs via soft BT loss.
        Call build_preference_dataset() first.
        Saves checkpoint to self.reward_ckpt_path.
        """
        assert self._preference_pairs, (
            "No preference pairs found. Call build_preference_dataset() first."
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rm     = RewardModel(self.reward_model_name).to(device)

        loader    = DataLoader(
            self._preference_pairs, batch_size=self.bt_batch_size,
            shuffle=True, collate_fn=self._collate_bt_batch,
        )
        optimizer = torch.optim.AdamW(rm.parameters(), lr=self.bt_lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(loader),
            num_training_steps=self.bt_epochs * len(loader),
        )

        for epoch in range(self.bt_epochs):
            epoch_loss = 0.0
            for enc_a, enc_b, p_a_wins in loader:
                enc_a    = {k: v.to(device) for k, v in enc_a.items()}
                enc_b    = {k: v.to(device) for k, v in enc_b.items()}
                p_a_wins = p_a_wins.to(device)
                loss = self._bt_loss(rm(**enc_a), rm(**enc_b), p_a_wins)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            print(f"[RewardModel] Epoch {epoch+1}/{self.bt_epochs} "
                  f"| Loss: {epoch_loss/len(loader):.4f}")

        Path(self.reward_ckpt_path).mkdir(parents=True, exist_ok=True)
        torch.save(rm.state_dict(), f"{self.reward_ckpt_path}/reward_model.pt")
        self.reward_tokenizer.save_pretrained(self.reward_ckpt_path)
        print(f"Reward model saved to {self.reward_ckpt_path}")

        rm.eval()
        self.reward_model = rm
        return rm

    def load_reward_model(self) -> RewardModel:
        """Load a previously trained reward model from disk (skip Phase 1)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rm = RewardModel(self.reward_model_name).to(device)
        rm.load_state_dict(torch.load(f"{self.reward_ckpt_path}/reward_model.pt"))
        rm.eval()
        self.reward_model = rm
        return rm

    def _score_with_reward_model(self, question: str, response: str) -> torch.Tensor:
        assert self.reward_model is not None, (
            "Call train_reward_model() or load_reward_model() first."
        )
        enc = self.reward_tokenizer(
            question, response, return_tensors="pt",
            truncation=True, max_length=self.bt_max_length,
        )
        device = next(self.reward_model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return self.reward_model(**enc)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — compute_loss override (GRPO-style unlearning)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Phase 2 — GRPO-style unlearning loss.

        Expected extra keys in inputs (added by your data collator):
          questions        : List[str]          raw question strings
          is_forget_sample : BoolTensor (batch,) which samples are from forget set

        Loss:
          total = pg_loss (forget samples via R(x,y)) 
                + grpo_beta * retain_loss (from super().compute_loss on retain samples)
        """
        assert self.reward_model is not None, (
            "Call train_reward_model() or load_reward_model() before trainer.train()."
        )

        questions        = inputs.pop("questions", None)
        is_forget_sample = inputs.pop("is_forget_sample", None)

        outputs = model(**inputs)
        logits  = outputs.logits

        # ── Forget: policy gradient with R(x,y) as reward ────────────────────
        if questions is not None and is_forget_sample is not None:
            forget_mask = is_forget_sample.bool()
            if forget_mask.any():
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"][forget_mask],
                        attention_mask=inputs["attention_mask"][forget_mask],
                        max_new_tokens=128,
                        do_sample=False,
                    )

                forget_questions = [q for q, m in zip(questions, forget_mask) if m]
                rewards = torch.stack([
                    self._score_with_reward_model(q, self.tokenizer.decode(g, skip_special_tokens=True))
                    for q, g in zip(forget_questions, generated_ids)
                ]).to(logits.device)

                # GRPO advantage: center rewards within the group
                advantage = rewards - rewards.mean()

                log_probs       = F.log_softmax(logits[forget_mask], dim=-1)
                labels          = inputs["labels"][forget_mask]
                token_log_probs = log_probs.gather(
                    2, labels.unsqueeze(-1).clamp(min=0)
                ).squeeze(-1)
                mask    = labels != -100
                pg_loss = -(advantage.unsqueeze(1) * token_log_probs * mask).sum() / mask.sum()
            else:
                pg_loss = torch.tensor(0.0, device=logits.device)
        else:
            pg_loss = torch.tensor(0.0, device=logits.device)

        # ── Retain: delegate to FinetuneTrainer CE/KL loss ───────────────────
        retain_loss, retain_outputs = super().compute_loss(
            model, inputs, return_outputs=True
        )

        total_loss = pg_loss + self.grpo_beta * retain_loss
        return (total_loss, outputs) if return_outputs else total_loss
```

---

## Usage

```python
from datasets import load_dataset
from trainer.reward_unlearn_trainer import RewardUnlearnTrainer

forget_split = load_dataset("locuslab/TOFU", "forget10")["train"]
retain_split = load_dataset("locuslab/TOFU", "retain90")["train"]

trainer = RewardUnlearnTrainer(
    model=model,
    args=training_args,
    train_dataset=forget_split,       # used by HF Trainer / Phase 2 loop
    eval_dataset=retain_split,
    tokenizer=tokenizer,
    data_collator=your_collator,
    # ── Phase 1 / 2 args ──────────────────────────────────────────────────
    forget_dataset=forget_split,      # used by Phase 1 pair collection
    judge_model=judge_model,
    judge_tokenizer=judge_tokenizer,
    reward_model_name="microsoft/deberta-v3-base",
    reward_ckpt_path="./ckpts/reward_model",
    grpo_beta=0.1,
)

# Phase 1 — run once
trainer.build_preference_dataset(cache_path="./cache/bt_pairs.jsonl")
trainer.train_reward_model()

# Phase 2 — unlearning
trainer.train()

# ── To skip Phase 1 on subsequent runs ─────────────────────────────────────
# trainer.load_reward_model()
# trainer.train()
```

---

## Data Collator Adapter

Your collator must inject two extra fields into each batch:

```python
def unlearn_collate_fn(batch):
    collated = default_data_collator(batch)
    collated["questions"]        = [b["question"] for b in batch]
    collated["is_forget_sample"] = torch.tensor(
        [b.get("is_forget", 1) for b in batch], dtype=torch.bool
    )
    return collated
```

---

## Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `verif_threshold` | 0.9 | Hard floor — V >= this -> reward = 0 |
| `bell_mu` | 0.2 | Peak of verifiability reward curve |
| `bell_sigma` | 0.15 | Width of reward peak |
| `bell_alpha` | 1.5 | Drift strength away from ground truth |
| `w_verify` | 0.4 | Weight of structural V signal |
| `w_judge` | 0.6 | Weight of behavioral J signal |
| `collapse_discount` | 0.1 | Multiplier for pure refusal responses |
| `grpo_beta` | 0.1 | Retain KL penalty coefficient in Phase 2 |
| `bt_lr` | 1e-5 | AdamW learning rate for reward model |
| `bt_epochs` | 3 | Reward model training epochs |