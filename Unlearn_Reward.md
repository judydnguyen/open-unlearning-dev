# =============================================================================
# Distilling a Generative Verifier into a Fast Classifier R_ψ
# Goal: transfer GPT-4-level semantic leakage judgment into a small,
#       fast discriminative model suitable for live GRPO reward calls.
# =============================================================================


# =============================================================================
# PHASE 1 — Offline Oracle Labeling (run once, before GRPO)
# =============================================================================
# The generative verifier (e.g. GPT-4) is used ONLY here — never during GRPO.
# It labels a large, diverse pool of (q, ŷ, x) triples with soft scores.

JUDGE_PROMPT = """
You are evaluating whether an answer reveals information about a specific entity.

Entity to forget: {x}
Question asked:   {q}
Model's answer:   {y_hat}

Does this answer reveal information about '{x}'?
Consider: direct naming, indirect description, associated works, implied identity,
circumlocution, pronoun references, or any fact that lets a reader identify '{x}'.

Respond with a single float between 0.0 and 1.0.
0.0 = answer fully reveals '{x}' (bad, leaks)
1.0 = answer reveals nothing about '{x}' (good, safe)
Only output the float. Nothing else.
"""

def build_distillation_dataset(
    theta_star,       # original model (pre-unlearning) — generates diverse outputs
    X_forget,         # forget targets
    Q_probe,          # probe questions per target
    judge_LLM,        # frozen GPT-4 or equivalent
    N_samples = 20,   # outputs to generate per (q, x) pair
):
    """
    For each (q, x) pair, sample N_samples outputs from theta_star at
    varying temperatures to get diversity across the leakage spectrum.
    Label each with the judge. Result: a rich soft-labeled dataset.
    """
    dataset = []

    for x in X_forget:
        for q in Q_probe[x]:

            # Sample at multiple temperatures to cover the output distribution:
            # - low T  → high-confidence, likely overtly leaky outputs
            # - mid T  → typical model behavior
            # - high T → diverse, sometimes evasive, sometimes creative leaks
            outputs = []
            for T in [0.3, 0.7, 1.0, 1.3]:
                for _ in range(N_samples // 4):
                    y = sample(theta_star, q, temperature=T)
                    outputs.append(y)

            # Also explicitly generate hard cases:

            # Hard negative: circumlocution that passes lexical phi but leaks
            y_circumlocution = judge_LLM(
                f"Rewrite this answer to avoid naming '{x}' directly "
                f"but preserve all the facts: {sample(theta_star, q)}"
            )
            outputs.append(y_circumlocution)

            # Hard positive: informative but genuinely safe answer
            y_safe_informative = judge_LLM(
                f"Answer '{q}' without revealing any information about '{x}'. "
                f"You may discuss the general topic but not '{x}' specifically."
            )
            outputs.append(y_safe_informative)

            # Label all outputs with the judge
            # Use multiple prompt templates and average — reduces prompt sensitivity
            for y_hat in outputs:
                scores = []
                for template in JUDGE_PROMPT_VARIANTS:   # 3–5 rephrasings
                    raw = judge_LLM(
                        template.format(x=x, q=q, y_hat=y_hat),
                        parse=float,
                        temperature=0.0    # deterministic judge
                    )
                    scores.append(clip(raw, 0.0, 1.0))

                soft_label = mean(scores)
                label_std  = std(scores)   # keep this — high std = ambiguous example

                dataset.append({
                    "q":          q,
                    "y_hat":      y_hat,
                    "x":          x,
                    "soft_label": soft_label,
                    "label_std":  label_std,   # used to weight loss later
                })

    return dataset   # typically 10k–100k examples depending on |X_forget|


# =============================================================================
# PHASE 2 — Dataset Quality Filtering
# =============================================================================
# Not all oracle labels are equally trustworthy.
# High label_std = judge was inconsistent across prompt templates = noisy label.
# Filter or downweight these before distillation.

def filter_dataset(dataset, std_threshold=0.15):
    """
    Split into clean and noisy subsets.
    Clean: judge was consistent → use with full weight.
    Noisy: judge disagreed across templates → downweight or discard.
    """
    clean = [d for d in dataset if d["label_std"] <= std_threshold]
    noisy = [d for d in dataset if d["label_std"] >  std_threshold]

    print(f"Clean: {len(clean)}, Noisy: {len(noisy)} "
          f"({100*len(noisy)/len(dataset):.1f}% discarded or downweighted)")

    # Optionally keep noisy examples with reduced weight
    for d in noisy:
        d["sample_weight"] = 1.0 - (d["label_std"] / 0.5)   # linearly decay weight

    return clean + noisy


# =============================================================================
# PHASE 3 — Train the Fast Classifier R_ψ via Distillation
# =============================================================================

# Architecture: small encoder LM + scalar head.
# Key choice: the input must include x (forget target) explicitly.
# Without conditioning on x, the model can't be target-aware.

# Input format: "Question: {q} [SEP] Answer: {y_hat} [SEP] Target: {x}"
# Output: scalar ∈ (0, 1)  — same semantics as the judge's soft label

def train_fast_classifier(dataset, base_encoder="deberta-v3-base"):

    model    = EncoderWithScalarHead(base_encoder)
    # Head: Linear(hidden_size → 1) + Sigmoid
    # Optionally: mean-pool over [CLS] + last 4 layers for richer representation

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataset) * N_epochs)

    for epoch in range(N_epochs):   # 3–5 epochs typically sufficient
        for batch in DataLoader(dataset, batch_size=32, shuffle=True):

            q_batch      = batch["q"]
            y_batch      = batch["y_hat"]
            x_batch      = batch["x"]
            labels       = batch["soft_label"]      # float in [0,1]
            weights      = batch.get("sample_weight", ones_like(labels))

            # Tokenize
            inputs = tokenize([
                f"Question: {q} [SEP] Answer: {y} [SEP] Target: {x}"
                for q, y, x in zip(q_batch, y_batch, x_batch)
            ])

            scores = model(inputs)   # shape (B,), values in (0,1)

            # --- Loss 1: KL distillation from soft labels ---
            # Treat labels as Bernoulli distributions and minimize KL.
            # This is softer than MSE and better preserves label uncertainty.
            loss_kl = mean(
                weights * (
                    labels * log(labels / (scores + 1e-8)) +
                    (1 - labels) * log((1 - labels) / (1 - scores + 1e-8))
                )
            )

            # --- Loss 2: Ranking loss on pairs within same (q, x) ---
            # For pairs (y_i, y_j) where label_i > label_j by margin,
            # enforce score_i > score_j. This is what GRPO actually needs.
            pairs = get_pairs_same_qx(batch)   # (i, j) where label_i > label_j + 0.1
            if pairs:
                s_i = scores[pairs.i]
                s_j = scores[pairs.j]
                loss_rank = mean(relu(s_j - s_i + margin))   # margin = 0.1

            # --- Loss 3: Focal term to hard examples ---
            # Down-weight easy examples (scores already near correct answer)
            # so gradient focuses on the decision boundary.
            gamma = 2.0
            p_t   = where(labels > 0.5, scores, 1 - scores)
            focal_weight = (1 - p_t) ** gamma
            loss_focal = mean(weights * focal_weight * BCE(scores, labels.round()))

            # Combine
            loss = loss_kl + λ_rank * loss_rank + λ_focal * loss_focal

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # --- Validation: ranking accuracy on held-out pairs ---
        # This is the metric that matters for GRPO, not classification accuracy
        rank_acc = evaluate_ranking_accuracy(model, val_dataset)
        print(f"Epoch {epoch}: ranking acc = {rank_acc:.4f}")

    return model


# =============================================================================
# PHASE 4 — Calibration
# =============================================================================
# The classifier's raw scores may be systematically over/under-confident.
# Calibrate so that score = 0.7 actually means "70% likely to be safe."
# This matters because GRPO advantages depend on the spread of rewards.

def calibrate_classifier(model, cal_dataset):
    """
    Temperature scaling: fit a single scalar T on a calibration set.
    p_cal = sigmoid(logit(p_raw) / T)
    T > 1 → spread scores out (model was overconfident)
    T < 1 → compress scores (model was underconfident)
    """
    # Collect raw logits (before sigmoid) and true soft labels
    logits = []
    labels = []
    with no_grad():
        for batch in DataLoader(cal_dataset, batch_size=64):
            logits.append(model.get_logits(batch))   # before sigmoid
            labels.append(batch["soft_label"])

    logits = concat(logits)
    labels = concat(labels)

    # Fit temperature by minimizing NLL on calibration set
    T = Parameter(tensor(1.0))
    optimizer = LBFGS([T], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        p_cal = sigmoid(logits / T)
        loss  = BCE(p_cal, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Measure Expected Calibration Error (ECE) before and after
    ece_before = compute_ece(sigmoid(logits), labels, n_bins=10)
    ece_after  = compute_ece(sigmoid(logits / T), labels, n_bins=10)
    print(f"ECE before: {ece_before:.4f} → after: {ece_after:.4f}, T = {T.item():.3f}")

    model.temperature = T.item()
    return model


# =============================================================================
# PHASE 5 — Validate Alignment with Judge Before Using in GRPO
# =============================================================================
# Critical: confirm the fast classifier agrees with the judge on cases
# that matter — specifically the hard boundary region (scores 0.3–0.7).
# Agreement on easy cases (score near 0 or 1) is trivial and not informative.

def validate_alignment(classifier, judge_LLM, theta_star, X_forget, Q_probe):

    hard_cases = []   # outputs where classifier score ∈ [0.3, 0.7]

    for x in X_forget:
        for q in Q_probe[x]:
            for T in [0.7, 1.0, 1.2]:
                y = sample(theta_star, q, temperature=T)
                score = classifier(q, y, x)
                if 0.3 <= score <= 0.7:
                    hard_cases.append((q, y, x, score))

    # Re-label hard cases with judge
    agreements = []
    for (q, y, x, clf_score) in hard_cases:
        judge_score = mean([
            judge_LLM(t.format(x=x, q=q, y_hat=y), parse=float, temperature=0.0)
            for t in JUDGE_PROMPT_VARIANTS
        ])
        # Agreement = are they on the same side of 0.5?
        same_side = (clf_score > 0.5) == (judge_score > 0.5)
        agreements.append(same_side)
        # Also measure correlation on continuous scores
        # (rank correlation matters more than binary agreement)

    boundary_accuracy   = mean(agreements)
    rank_correlation    = spearman_r(
        [s for _,_,_,s in hard_cases],
        [judge_score for ... in hard_cases]   # from re-labeling above
    )

    print(f"Boundary accuracy: {boundary_accuracy:.3f}")
    print(f"Rank correlation with judge: {rank_correlation:.3f}")

    # Rule of thumb: only proceed to GRPO if:
    # boundary_accuracy > 0.80 AND rank_correlation > 0.75
    assert boundary_accuracy > 0.80, "Classifier too uncertain on hard cases"
    assert rank_correlation  > 0.75, "Ranking unreliable — distillation failed"

    return boundary_accuracy, rank_correlation


# =============================================================================
# PHASE 6 — Drop-in Replacement in GRPO
# =============================================================================
# The fast classifier now replaces both phi (lexical) and the generative
# verifier in the GRPO reward call. One forward pass per output, ~5ms.

def grpo_reward(q, y_hat, x_forget, classifier):
    """
    Single forward pass through the distilled classifier.
    Returns scalar reward ∈ (0, 1) for use in GRPO advantage computation.
    Calibrated temperature is applied inside classifier.forward().
    """
    inp   = tokenize(f"Question: {q} [SEP] Answer: {y_hat} [SEP] Target: {x_forget}")
    logit = classifier.encoder(inp).pooled   # single forward pass
    score = sigmoid(logit / classifier.temperature)
    return score.item()


# =============================================================================
# PUTTING IT ALL TOGETHER — Training Schedule
# =============================================================================

def full_pipeline(theta_star, X_forget, Q_probe, judge_LLM, D_r):

    # Step 1: One-time offline oracle labeling (~hours, done once)
    raw_dataset = build_distillation_dataset(theta_star, X_forget, Q_probe, judge_LLM)
    dataset     = filter_dataset(raw_dataset)

    # Step 2: Train fast classifier (~minutes on 1 GPU)
    classifier  = train_fast_classifier(dataset)

    # Step 3: Calibrate (~seconds)
    classifier  = calibrate_classifier(classifier, cal_dataset=dataset[-1000:])

    # Step 4: Validate alignment with judge on hard cases
    # If this fails, go back and add more hard negatives to dataset
    b_acc, r_corr = validate_alignment(classifier, judge_LLM, theta_star, X_forget, Q_probe)

    # Step 5: GRPO unlearning — classifier is the live reward signal
    # No judge calls during this phase. Classifier is FROZEN.
    theta_prime = purge_grpo(
        theta_star  = theta_star,
        reward_fn   = lambda q, y, x: grpo_reward(q, y, x, classifier),
        X_forget    = X_forget,
        Q_probe     = Q_probe,
        D_r         = D_r,
    )

    # Step 6: Final audit with judge (not classifier) — trust but verify
    final_scores = []
    for x in X_forget:
        for q in Q_probe[x]:
            y = sample(theta_prime, q)
            judge_score = mean([
                judge_LLM(t.format(x=x, q=q, y_hat=y), parse=float, temperature=0.0)
                for t in JUDGE_PROMPT_VARIANTS
            ])
            final_scores.append(judge_score)

    print(f"Final semantic safety (judge): {mean(final_scores):.3f}")
    # Compare this to classifier's score on same outputs — check for drift
    clf_scores = [grpo_reward(q, y, x, classifier) for ...]
    drift = mean(abs(array(final_scores) - array(clf_scores)))
    print(f"Classifier drift from judge post-unlearning: {drift:.3f}")
    # If drift > 0.1, the classifier's training distribution has shifted
    # and its rewards during GRPO may have been unreliable in later steps.

    return theta_prime, classifier


# =============================================================================
# KEY HYPERPARAMETERS AND RULES OF THUMB
# =============================================================================
#
# Dataset size:
#   - Minimum: 500 labeled examples per forget target
#   - Sweet spot: 2000–5000 per target
#   - Diminishing returns beyond 10k per target
#
# Hard negatives ratio:
#   - At least 30% of training data should be hard negatives
#   - (circumlocutions, implicit references, pronoun-heavy outputs)
#   - Too few → classifier fails on exactly the cases GRPO needs it for
#
# Temperature scaling:
#   - Expected T ∈ [0.8, 1.5] for a well-trained DeBERTa classifier
#   - T >> 1.5 suggests overconfident training → add label smoothing
#   - T << 0.8 suggests underconfident training → reduce dropout
#
# Ranking accuracy threshold (Phase 5):
#   - < 0.75 → not safe for GRPO. Add more contrastive pairs, retrain.
#   - 0.75–0.85 → acceptable, monitor drift in Phase 6
#   - > 0.85 → good to go
#
# Classifier inference cost:
#   - DeBERTa-base: ~3–5ms per (q, ŷ, x) on A100
#   - With group size W=8: ~24–40ms per prompt — negligible vs. policy forward pass
#   - Compare to generative verifier: ~200–500ms per call (10–100x slower)