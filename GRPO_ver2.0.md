# GRPO-Based Machine Unlearning — Claude Code Instruction Guideline

## Overview

This guideline instructs you (Claude Code) to implement **GRPO-based machine unlearning** for LLMs. The goal is to make a model forget a specific set of knowledge (the *forget set*) while retaining performance on everything else (the *retain set*), using Group Relative Policy Optimization (GRPO) with rule-based rewards — no learned reward model required.

Read this entire document before writing any code.

---

## 1. Conceptual Framework

### What is GRPO?
GRPO replaces the critic/value network in PPO with **group-relative advantage estimation**. For each prompt, sample a group of `G` responses, compute rewards for each, then normalize advantages within the group:

```
A(o_i) = (r(o_i) - mean(r)) / std(r)
```

This relative signal is sufficient for stable policy optimization without a value function.

### What is Unlearning here?
Unlearning = suppressing the model's ability to output or represent specific knowledge, while preserving general utility. This is an **output-level** problem (and ideally a **representation-level** problem too).

### The Core Tension
The model can always cheat by refusing everything. Your reward design **must** penalize over-refusal explicitly.

---

## 2. Data Setup

### 2.1 Forget Set
- A corpus of documents/facts the model should forget
- Extract named entities using NER (e.g., spaCy `en_core_web_trf`)
- Build a **forbidden entity list** and a **forbidden fact list**
- Generate forget prompts: questions that would elicit the forgotten knowledge

```python
import spacy
nlp = spacy.load("en_core_web_trf")

def extract_forbidden_entities(forget_corpus: list[str]) -> set[str]:
    entities = set()
    for doc_text in forget_corpus:
        doc = nlp(doc_text)
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "WORK_OF_ART"}:
                entities.add(ent.text.lower())
    return entities
```

### 2.2 Retain Set
- A corpus of general knowledge/tasks the model should still perform well on
- Include a diverse mix: QA, reasoning, factual recall (unrelated to forget set)

### 2.3 Prompt Construction
For each training step, sample a **mixed batch**:
- ~50% forget prompts → model should express ignorance / safe refusal
- ~50% retain prompts → model should answer correctly

---

## 3. Reward Design

Use a **composite reward** with four components. All rewards are in range `[-1, 1]` or `[0, 1]` and weighted by tunable coefficients.

### 3.1 Forget Reward (NER-based, graduated)

Unlike PURGE's binary 0/1, use a **soft graduated penalty** based on entity salience:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_forget_reward(
    response: str,
    forbidden_entities: set[str],
    entity_salience: dict[str, float]  # TF-IDF scores
) -> float:
    response_lower = response.lower()
    found = [e for e in forbidden_entities if e in response_lower]
    if not found:
        return 1.0
    penalty = sum(entity_salience.get(e, 0.5) for e in found)
    return max(-1.0, 1.0 - penalty)
```

### 3.2 Semantic Leakage Reward

Catch paraphrase/synonym leakage that NER misses:

```python
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-mpnet-base-v2")
forget_embeddings = embedder.encode(forget_corpus)  # precompute

def compute_semantic_reward(response: str, forget_embeddings, threshold=0.7) -> float:
    resp_emb = embedder.encode(response)
    max_sim = max(util.cos_sim(resp_emb, fe).item() for fe in forget_embeddings)
    if max_sim > threshold:
        return -1.0
    return 1.0 - max_sim  # partial penalty below threshold
```

### 3.3 Retain Reward

Standard accuracy reward on retain prompts:

```python
def compute_retain_reward(response: str, ground_truth: str) -> float:
    # Exact match for factual QA
    if ground_truth.lower().strip() in response.lower():
        return 1.0
    # Optionally use a lightweight judge model for open-ended retain tasks
    return 0.0
```

### 3.4 Over-Refusal Penalty

Prevent the model from refusing everything:

```python
REFUSAL_PHRASES = [
    "i don't know", "i cannot", "i'm not sure", "i have no information",
    "i can't answer", "i don't have access"
]

def compute_over_refusal_penalty(response: str, prompt_type: str) -> float:
    """
    Only penalize refusals on RETAIN prompts.
    Refusals on FORGET prompts are rewarded via forget_reward.
    """
    if prompt_type == "retain":
        is_refusal = any(p in response.lower() for p in REFUSAL_PHRASES)
        return -1.0 if is_refusal else 0.0
    return 0.0
```

### 3.5 Composite Reward Function

```python
def compute_reward(
    response: str,
    prompt_type: str,       # "forget" or "retain"
    ground_truth: str,
    forbidden_entities: set[str],
    entity_salience: dict[str, float],
    forget_embeddings,
    alpha=0.4, beta=0.4, gamma=0.1, delta=0.1
) -> float:
    if prompt_type == "forget":
        r_forget = compute_forget_reward(response, forbidden_entities, entity_salience)
        r_semantic = compute_semantic_reward(response, forget_embeddings)
        return alpha * r_forget + beta * r_semantic
    else:  # retain
        r_retain = compute_retain_reward(response, ground_truth)
        r_over_refusal = compute_over_refusal_penalty(response, prompt_type)
        return gamma * r_retain + delta * r_over_refusal
```

> **Tuning note:** Start with `alpha=0.4, beta=0.4, gamma=0.1, delta=0.1`. If retain performance degrades, increase `gamma`. If the model still leaks on paraphrases, increase `beta`.

---

## 4. GRPO Training Loop

### 4.1 Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-base-model")  # frozen reference
```

### 4.2 Config

```python
config = GRPOConfig(
    num_generations=8,          # group size G — sample 8 responses per prompt
    max_new_tokens=256,
    temperature=0.9,            # keep entropy high early in training
    learning_rate=1e-6,         # low LR — unlearning is sensitive
    kl_coef=0.1,                # KL penalty against reference model — critical for retain
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    output_dir="./grpo_unlearn_output",
)
```

### 4.3 Reward Wrapper for TRL

```python
def reward_fn(prompts, responses, **kwargs):
    rewards = []
    for prompt, response in zip(prompts, responses):
        prompt_type = kwargs["prompt_types"][prompts.index(prompt)]
        ground_truth = kwargs["ground_truths"][prompts.index(prompt)]
        r = compute_reward(
            response=response,
            prompt_type=prompt_type,
            ground_truth=ground_truth,
            forbidden_entities=forbidden_entities,
            entity_salience=entity_salience,
            forget_embeddings=forget_embeddings,
        )
        rewards.append(r)
    return rewards
```

### 4.4 Trainer

```python
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    train_dataset=mixed_dataset,  # forget + retain prompts
    reward_funcs=reward_fn,
    tokenizer=tokenizer,
)
trainer.train()
```

---

## 5. Optional: Self-Check Block in Reasoning Trace

To encourage the model to internally audit its response, prompt it with a thinking template:

```python
FORGET_SYSTEM_PROMPT = """
You are a helpful assistant. Before answering, check:
<think>
Does my response reveal any specific information about [TOPIC]?
If yes → revise to express appropriate ignorance.
If no → proceed with the answer.
</think>
"""
```

Add a **format reward** that checks:
1. `<think>` block is present
2. The conclusion in `<think>` is consistent with the final answer (no leaking after saying "no")

```python
def compute_format_reward(response: str) -> float:
    has_think = "<think>" in response and "</think>" in response
    if not has_think:
        return 0.0
    think_content = response.split("<think>")[1].split("</think>")[0]
    answer_content = response.split("</think>")[-1]
    # Penalize if think says "no leak" but answer contains forbidden entities
    think_says_safe = "if no" in think_content.lower() or "proceed" in think_content.lower()
    answer_leaks = any(e in answer_content.lower() for e in forbidden_entities)
    if think_says_safe and answer_leaks:
        return -1.0  # inconsistency penalty
    return 1.0 if has_think else 0.0
```

---

## 6. Evaluation

After training, evaluate on three axes:

| Metric | What it measures | Tool |
|---|---|---|
| **Forget Accuracy** | Does model still answer forget prompts correctly? (want ~0%) | Exact match / LLM judge |
| **Retain Accuracy** | Does model still answer retain prompts correctly? (want ~100%) | Exact match |
| **Semantic Leakage Score** | Cosine similarity of responses to forget corpus | SentenceTransformers |
| **Over-refusal Rate** | % of retain prompts refused | Rule-based |
| **ROUGE-L vs forget** | N-gram overlap with forget documents | `rouge_score` library |

Use the **TOFU benchmark** if available for standardized evaluation.

---

## 7. Common Failure Modes & Fixes

| Failure | Symptom | Fix |
|---|---|---|
| Over-refusal | Model refuses retain questions | Increase `gamma` (retain reward weight) |
| Surface suppression only | Model says "I don't know" but hidden states still encode forget knowledge | Add probing-based reward (see below) |
| Paraphrase leakage | Model avoids entity names but describes forgotten content | Increase `beta` (semantic reward weight) |
| Reward collapse | All responses get ~same reward, no learning signal | Increase group size G, check reward variance |
| KL divergence too high | Retain performance collapses | Reduce `learning_rate` or increase `kl_coef` |

### Optional: Probing-Based Reward (Advanced)

If you want to target representational unlearning, not just output suppression:

```python
# Train a linear probe on hidden states to detect forget knowledge
# Then use probe confidence as an additional penalty signal
def compute_probe_reward(hidden_states, probe_model) -> float:
    prob_forget = probe_model.predict_proba(hidden_states)[:, 1].mean()
    return -prob_forget  # penalize if hidden states encode forget knowledge
```

This requires a pre-trained probe — train it on the original model before GRPO starts.

---

## 8. File Structure

```
grpo_unlearning/
├── data/
│   ├── forget_corpus.jsonl
│   ├── retain_corpus.jsonl
│   └── mixed_dataset.py       # builds mixed prompt dataset
├── rewards/
│   ├── ner_reward.py           # graduated NER-based forget reward
│   ├── semantic_reward.py      # cosine similarity leakage reward
│   ├── retain_reward.py        # accuracy reward for retain set
│   ├── format_reward.py        # self-check block reward
│   └── composite.py            # combines all rewards
├── train.py                    # GRPO training loop
├── evaluate.py                 # evaluation on all axes
└── GUIDELINE.md                # this file
```

---

## 9. Key References

- **PURGE** (2025): Reinforcement Unlearning via GRPO — binary NER reward baseline
- **DeepSeek-R1** (2025): Rule-based GRPO with format + accuracy rewards
- **TOFU Benchmark** (2024): Standard unlearning evaluation for LLMs
- **CE-U** (2025): Cross Entropy Unlearning — alternative loss for stable unlearning

---

*End of guideline. Implement in the order: data setup → reward functions → training loop → evaluation.*