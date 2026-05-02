import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from trainer.unlearn.grad_diff import GradDiff


class PerSampleEncoder(nn.Module):
    """Residual MLP encoder that produces a steering direction.

    forward: r = alpha * h_pooled + (1 - alpha) * delta(h_pooled)
    Last layer of delta is zero-initialized so r = alpha * h_pooled at init,
    a meaningful (non-zero, non-NaN) starting direction.
    """
    def __init__(self, hidden_size: int, latent_dim: int = 256, num_layers: int = 2, alpha: float = 0.2):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        self.alpha = alpha
        layers = [nn.Linear(hidden_size, latent_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(latent_dim, latent_dim), nn.GELU()]
        layers.append(nn.Linear(latent_dim, hidden_size))
        self.net = nn.Sequential(*layers)

        # In __init__, REMOVE the nn.init.zeros_ lines, OR replace with:
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        delta = self.net(h_pooled)
        return self.alpha * h_pooled + (1 - self.alpha) * delta
        # return delta


class LatentRMU(GradDiff):
    def __init__(
        self,
        module_regex=r"model\.layers\.7",
        trainable_params_regex=(r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight",),
        steering_coeff=20,
        latent_dim=256,
        encoder_epochs=2,
        encoder_layers=2,
        encoder_alpha=0.5,
        orth_weight=1.0,
        anchor_weight=1.0,
        encoder_lr=1e-3,
        forget_warmup_steps=0,
        coeff_warmup_steps=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        self.trainable_params_regex = list(trainable_params_regex)
        self.module_regex = module_regex
        self.model_module = self._get_matching_module(self.model, self.module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, self.module_regex)
        self.steering_coeff = steering_coeff
        self.encoder_epochs = encoder_epochs
        self.orth_weight = orth_weight
        self.anchor_weight = anchor_weight
        self.encoder_lr = encoder_lr
        self.forget_warmup_steps = forget_warmup_steps
        self.coeff_warmup_steps = coeff_warmup_steps
        self._phase = 1
        self._phase2_step = 0

        hidden_size = self.model.config.hidden_size
        self.encoder = PerSampleEncoder(hidden_size, latent_dim, encoder_layers, encoder_alpha)
        self.encoder.to(next(self.model.parameters()).device)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_dontknow_activations(self, forget_inputs):
        input_ids = forget_inputs["input_ids"].clone()
        labels = forget_inputs["labels"]
        answer_mask = labels != -100
        input_ids[answer_mask] = self.tokenizer.unk_token_id or self.tokenizer.pad_token_id
        proxy_inputs = {
            "input_ids": input_ids,
            "attention_mask": forget_inputs["attention_mask"],
        }
        with torch.no_grad():
            h, _ = self.forward_with_cache(
                self.ref_model, proxy_inputs, self.ref_module, no_grad=True
            )
        return F.normalize(h.float().mean(dim=1), dim=-1)

    def _get_matching_module(self, model, module_regex):
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module
        matched = {n: m for n, m in model.named_modules() if re.fullmatch(module_regex, n)}
        if len(matched) != 1:
            raise ValueError(f"Expected 1 module match for {module_regex}, got {len(matched)}")
        return next(iter(matched.values()))

    def _freeze_all_params(self, model, requires_grad=False):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        for name, param in model.named_parameters():
            if any(re.fullmatch(pat, name) for pat in trainable_params_regex):
                param.requires_grad = requires_grad

    def _get_phase2_params(self):
        return [
            p for n, p in self.model.named_parameters()
            if any(re.fullmatch(pat, n) for pat in self.trainable_params_regex)
            and p.requires_grad
        ]

    def forward_with_cache(self, model, inputs, module, no_grad=True):
        cache = []
        def hook(_m, _i, output):
            cache.append(output[0] if isinstance(output, tuple) else output)
        handle = module.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(not no_grad):
                outputs = model(**inputs)
        finally:
            handle.remove()
        return cache[0], outputs

    def compute_activation_loss(self, activation1, activation2, mask):
        squared_diff = F.mse_loss(activation1, activation2, reduction="none")
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
        squared_diff_sum = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)
        num_tokens = mask.sum(dim=-1, keepdim=True)
        return (squared_diff_sum / num_tokens).mean()

    # ------------------------------------------------------------------ #
    #  Two-phase training                                                  #
    # ------------------------------------------------------------------ #

    def evaluate(self, *args, **kwargs):
        if self._phase == 1:
            return {}
        return super().evaluate(*args, **kwargs)

    def train(self, resume_from_checkpoint=None, **kwargs):
        original_epochs = self.args.num_train_epochs

        self._phase = 1
        self._freeze_all_params(self.model, requires_grad=False)
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        self.args.num_train_epochs = self.encoder_epochs
        super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        self._phase = 2
        self._freeze_all_params(self.encoder, requires_grad=False)
        self.encoder.eval()
        self.optimizer = None
        self.lr_scheduler = None
        self.args.num_train_epochs = original_epochs - self.encoder_epochs
        super().train(**kwargs)

        self.args.num_train_epochs = original_epochs

    def create_optimizer(self):
        if self._phase == 1:
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            optimizer_cls, _ = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(
                self.encoder.parameters(),
                lr=self.encoder_lr,
            )
        else:
            self._freeze_all_params(self.encoder, requires_grad=False)
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            super().create_optimizer()

    # ------------------------------------------------------------------ #
    #  Steering vector — single source of truth, used by BOTH phases     #
    # ------------------------------------------------------------------ #

    def _ramped_steering_coeff(self) -> float:
        if self.coeff_warmup_steps <= 0:
            return self.steering_coeff
        return self.steering_coeff * max(0.0, 1.0 - self._phase2_step / self.coeff_warmup_steps)

    def _compute_steering(self, h_forget: torch.Tensor, steering_coeff: float):
        """Returns (r_normed, r_scaled_unsqueezed).

        r_normed: (B, H), unit-norm encoder direction. Carries gradient through encoder.
        r_scaled_unsqueezed: (B, 1, H), ready to broadcast across sequence positions.
        """
        pooled = h_forget.float().mean(dim=1)
        r = self.encoder(pooled)
        r_normed = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
        r_scaled = r_normed.unsqueeze(1).to(h_forget.dtype) * steering_coeff
        return r_normed, r_scaled

    def _build_steering_target(self, h_forget_ref: torch.Tensor, r_scaled: torch.Tensor) -> torch.Tensor:
        """The target Phase 2 actually pulls toward, and Phase 1 simulates against.

        target = h_forget_ref + steering_coeff * r_normed
        """
        return h_forget_ref + r_scaled

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = {k: inputs["forget"][k] for k in ("input_ids", "attention_mask", "labels")}
        retain_inputs = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        mask = forget_inputs["labels"] != -100

        if self._phase == 1:
            return self._phase1_loss(forget_inputs, retain_inputs, mask)

        loss, forget_outputs = self._phase2_loss(model, forget_inputs, retain_inputs, mask)
        return (loss, forget_outputs) if return_outputs else loss

    # ---- Phase 1 ----------------------------------------------------- #

    def _phase1_loss(self, forget_inputs, retain_inputs, mask):
        # Forward through model (gradient flows for conflict computation)
        h_forget, _ = self.forward_with_cache(
            self.model, forget_inputs, self.model_module, no_grad=False
        )
        # Forward through ref (no grad — used as fixed baseline)
        with torch.no_grad():
            h_forget_ref, _ = self.forward_with_cache(
                self.ref_model, forget_inputs, self.ref_module, no_grad=True
            )
        h_forget_ref = h_forget_ref.to(h_forget.dtype)

        # Encoder direction (gradient flows through r_normed)
        r_normed, r_scaled = self._compute_steering(h_forget, self.steering_coeff)

        # SAME target Phase 2 will use:
        target = self._build_steering_target(h_forget_ref, r_scaled)

        # Detach target from MODEL's graph (we want gradient to flow back through
        # h_forget — the model — and through r_scaled — the encoder — but NOT
        # treat target as a moving goal during this MSE).
        # Actually we want gradient through r_scaled (encoder), so we don't fully
        # detach. Just detach h_forget_ref (already no_grad). r_scaled keeps grad.
        forget_loss = self.compute_activation_loss(
            h_forget.float(), target.float(), mask
        )

        # Retain NLL — gradient is non-zero (phase2 params have requires_grad)
        retain_loss_for_conflict = self.model(**retain_inputs).loss

        phase2_params = self._get_phase2_params()

        grad_forget = torch.autograd.grad(
            forget_loss, phase2_params,
            create_graph=True, allow_unused=True, retain_graph=True,
        )
        grad_retain = torch.autograd.grad(
            retain_loss_for_conflict, phase2_params,
            create_graph=False, allow_unused=True,
        )
        grad_retain = [g.detach() if g is not None else None for g in grad_retain]

        paired = [(a, b) for a, b in zip(grad_forget, grad_retain) if a is not None and b is not None]

        # IDK anchor: pull r_normed toward dontknow direction
        dontknow_dirs = self._get_dontknow_activations(forget_inputs)
        anchor_loss = 1 - (r_normed * dontknow_dirs).sum(dim=-1).mean()

        if not paired:
            grad_conflict = torch.tensor(0.0, device=h_forget.device, requires_grad=True)
            cos_sim_val = 0.0
            gf_norm = gr_norm = 0.0
        else:
            g1 = torch.cat([a.flatten() for a, _ in paired])
            g2 = torch.cat([b.flatten() for _, b in paired])
            cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).squeeze()
            grad_conflict = cos_sim ** 2
            cos_sim_val = cos_sim.item()
            gf_norm = g1.norm().item()
            gr_norm = g2.norm().item()

        loss = (
            self.orth_weight * grad_conflict
            + self.anchor_weight * anchor_loss
        )

        # Diagnostic: how far is target from h_forget? Should be ~steering_coeff at init.
        with torch.no_grad():
            displacement = (target - h_forget).norm(dim=-1).mean().item()

        self.log({
            "phase1/conflict_cos": cos_sim_val,
            "phase1/conflict_cos2": grad_conflict.item(),
            "phase1/anchor_loss": anchor_loss.item(),
            "phase1/forget_loss_sim": forget_loss.item(),
            "phase1/retain_nll": retain_loss_for_conflict.item(),
            "phase1/r_norm_pre_normalize": (r_normed * (r_normed.norm(dim=-1, keepdim=True))).norm(dim=-1).mean().item(),
            "phase1/displacement_h_to_target": displacement,
            "phase1/g_forget_norm": gf_norm,
            "phase1/g_retain_norm": gr_norm,
        })
        return loss

    # ---- Phase 2 ----------------------------------------------------- #

    def _phase2_loss(self, model, forget_inputs, retain_inputs, mask):
        h_forget, forget_outputs = self.forward_with_cache(
            model, forget_inputs, self.model_module, no_grad=False
        )
        with torch.no_grad():
            h_forget_ref, _ = self.forward_with_cache(
                self.ref_model, forget_inputs, self.ref_module, no_grad=True
            )
            # Same target construction as Phase 1
            _, r_scaled = self._compute_steering(h_forget, self._ramped_steering_coeff())
            target = self._build_steering_target(h_forget_ref, r_scaled)

        target = target.expand_as(h_forget)
        forget_loss = self.compute_activation_loss(h_forget, target, mask)
        retain_loss = self.compute_retain_loss(model, retain_inputs)

        if self.forget_warmup_steps > 0:
            warmup_coeff = min(1.0, self._phase2_step / self.forget_warmup_steps)
        else:
            warmup_coeff = 1.0
        self._phase2_step += 1

        loss = self.gamma * warmup_coeff * forget_loss + self.alpha * retain_loss

        self.log({
            "train/forget_loss": forget_loss.item(),
            "train/retain_loss": retain_loss.item(),
            "train/forget_warmup_coeff": warmup_coeff,
            "train/steering_coeff": self._ramped_steering_coeff(),
            "train/displacement": (target[:, :1, :] - h_forget_ref).norm(dim=-1).mean().item(),
        })
        return loss, forget_outputs

    # ------------------------------------------------------------------ #

    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "EMBED_DIFF":
            model_retain_act, _ = self.forward_with_cache(
                model, retain_inputs, self.model_module, no_grad=False
            )
            with torch.no_grad():
                ref_retain_act, _ = self.forward_with_cache(
                    self.ref_model, retain_inputs, self.ref_module, no_grad=True
                )
            mask = retain_inputs["labels"] != -100
            return self.compute_activation_loss(
                model_retain_act, ref_retain_act.to(model_retain_act.device), mask,
            )
        return super().compute_retain_loss(model, retain_inputs)