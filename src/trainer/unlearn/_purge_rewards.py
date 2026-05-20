"""
PURGE reward components (strzar/purge).

Three per-completion reward functions over a set of forget terms:

  binary             1.0 if no forget term is present, 0.0 otherwise.
  exponential_decay  base ^ (-count / tau); smooth analogue of binary.
  pagerank           1 - sum(PPR-weighted penalties of matched terms),
                     clamped to [0, 1].

Plus helpers used by SteerGRPO when ``reward_type`` is one of the three.
Keep this module free of Trainer imports so SteerGRPO can use it without
introducing a circular dependency with PurgeGRPO.

Reference: https://github.com/strzar/purge
"""

import json
import math
import re
from typing import List, Optional

import numpy as np
import torch


# ── Reward functions ───────────────────────────────────────────────────────

def reward_binary(
    completions: List[str], *, pattern: Optional[re.Pattern], **_,
) -> List[float]:
    if pattern is None:
        return [1.0] * len(completions)
    return [0.0 if pattern.search(c) else 1.0 for c in completions]


def reward_exponential_decay(
    completions: List[str], *, pattern: Optional[re.Pattern],
    decay_tau: float = 1.0, decay_base: float = math.e, **_,
) -> List[float]:
    if pattern is None:
        return [1.0] * len(completions)
    tau = max(decay_tau, 1e-8)
    return [float(decay_base ** (-len(pattern.findall(c)) / tau)) for c in completions]


def reward_pagerank(
    completions: List[str], *, terms: List[str], weights: Optional[np.ndarray], **_,
) -> List[float]:
    if not terms or weights is None:
        return [1.0] * len(completions)
    pats = [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in terms]
    out = []
    for c in completions:
        penalty = sum(
            float(w) for pat, w in zip(pats, weights) if pat.search(c)
        )
        out.append(float(max(0.0, min(1.0, 1.0 - penalty))))
    return out


REWARD_FUNCS = {
    "binary": reward_binary,
    "exponential_decay": reward_exponential_decay,
    "pagerank": reward_pagerank,
}


def is_purge_reward_type(reward_type: str) -> bool:
    return reward_type in REWARD_FUNCS


# ── Helpers ────────────────────────────────────────────────────────────────

def resolve_terms(
    forget_words: Optional[List[str]],
    forget_words_file: Optional[str],
    target_entity: Optional[str],
) -> List[str]:
    if forget_words:
        terms = list(forget_words)
    elif forget_words_file:
        with open(forget_words_file) as f:
            terms = json.load(f)
        if not isinstance(terms, list) or not all(isinstance(x, str) for x in terms):
            raise ValueError(
                f"forget_words_file must be a JSON array of strings: {forget_words_file}"
            )
    else:
        terms = []
    if target_entity:
        terms = [target_entity] + [t for t in terms if t != target_entity]
    seen, ordered = set(), []
    for t in terms:
        k = t.lower()
        if k not in seen and t.strip():
            seen.add(k)
            ordered.append(t)
    return ordered


def compile_pattern(terms: List[str]) -> Optional[re.Pattern]:
    if not terms:
        return None
    return re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b",
        re.IGNORECASE,
    )


def compute_pagerank_weights(
    terms: List[str], model, tokenizer,
    quantile: float = 0.75, seeded: bool = False,
) -> np.ndarray:
    """Cosine-similarity graph → top (1 - quantile) edges → personalized PPR."""
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("pagerank reward needs networkx") from e
    if len(terms) <= 1:
        return np.ones(len(terms), dtype=np.float64)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    embs = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for t in terms:
                enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=32).to(device)
                out = model(
                    **enc, output_hidden_states=True, use_cache=False, return_dict=True,
                )
                h = out.hidden_states[-1]
                mask = enc.attention_mask.unsqueeze(-1).float()
                pooled = ((h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0))
                embs.append(pooled[0].float().cpu().numpy())
    finally:
        if was_training:
            model.train()
    emb = np.stack(embs)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = emb @ emb.T
    np.fill_diagonal(sim, 0.0)
    adj = (sim > np.quantile(sim, float(np.clip(quantile, 0.0, 1.0)))).astype(np.float64)
    adj = np.maximum(adj, adj.T)
    G = nx.from_numpy_array(adj)
    pers = {0: 1.0, **{i: 0.0 for i in range(1, len(terms))}} if seeded else None
    pr = nx.pagerank(G, personalization=pers, alpha=0.85)
    w = np.array([pr.get(i, 0.0) for i in range(len(terms))], dtype=np.float64)
    if w.max() > 0:
        w /= w.max()
    return w
