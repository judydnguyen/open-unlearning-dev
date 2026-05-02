import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from trainer.unlearn.grad_diff import GradDiff


# ------------------------------------------------------------------ #
#  Distributed helpers                                               #
# ------------------------------------------------------------------ #

def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _world_size():
    return dist.get_world_size() if _is_dist() else 1


def _rank():
    return dist.get_rank() if _is_dist() else 0


def _all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce + average. For non-grad tensors only."""
    if not _is_dist():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(_world_size())
    return tensor


# ------------------------------------------------------------------ #
#  Encoder                                                            #
# ------------------------------------------------------------------ #

class PerSampleEncoder(nn.Module):
    """Residual MLP encoder that produces a steering direction.

    forward: r = alpha * h_pooled + (1 - alpha) * delta(h_pooled)
    """
    def __init__(self, hidden_size: int, latent_dim: int = 256,
                 num_layers: int = 2, alpha: float = 0.2):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        self.alpha = alpha
        layers = [nn.Linear(hidden_size, latent_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(latent_dim, latent_dim), nn.GELU()]
        layers.append(nn.Linear(latent_dim, hidden_size))
        self.net = nn.Sequential(*layers)

        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        delta = self.net(h_pooled)
        return self.alpha * h_pooled + (1 - self.alpha) * delta


# ------------------------------------------------------------------ #
#  Trainer                                                            #
# ------------------------------------------------------------------ #

class LatentRMUParallel(GradDiff):
    """LatentRMU adapted for distributed (DDP) training.

    Differences from single-GPU LatentRMU:
      - Encoder is wrapped in DDP so its gradients are all-reduced across
        ranks during Phase 1's loss.backward().
      - Module-pattern matching unwraps DDP-wrapped models correctly.
      - Phase 1 logs are reduced across ranks for clean monitoring.
      - Conflict cosine is computed per-rank (see note in _phase1_loss).
    """

    def __init__(
        self,
        module_regex=r"model\.layers\.7",
        trainable_params_regex=(r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight",),
        steering_coeff=20,
        latent_dim=256,
        encoder_epochs=2,
        encoder_layers=2,
        encoder_alpha=0.2,
        orth_weight=1.0,
        anchor_weight=1.0,
        encoder_lr=1e-3,
        forget_warmup_steps=0,
        coeff_warmup_steps=0,
        *args, **kwargs,
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
        device = next(self.model.parameters()).device

        # Raw encoder lives here. DDP wrapper is created lazily once dist
        # is initialized (HF Trainer initializes dist inside train()).
        self._encoder_raw = PerSampleEncoder(
            hidden_size, latent_dim, encoder_layers, encoder_alpha
        )
        self._encoder_raw.to(device)
        self.encoder = self._encoder_raw  # replaced by DDP wrapper after init
        self._encoder_ddp_wrapped = False

    # ------------------------------------------------------------------ #
    #  DDP setup                                                          #
    # ------------------------------------------------------------------ #

    def _maybe_wrap_encoder(self):
        """Idempotent DDP wrapping. Call before encoder is used in Phase 1."""
        if self._encoder_ddp_wrapped or not _is_dist():
            return

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        self._encoder_raw.to(device)

        self.encoder = DDP(
            self._encoder_raw,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        self._encoder_ddp_wrapped = True

    def _encoder_module(self) -> nn.Module:
        """Return the underlying (un-DDP) encoder for state access / freezing."""
        return self._encoder_raw

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unwrap(model):
        """Strip DDP / DeepSpeed wrappers to expose the underlying module."""
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module
        if isinstance(model, DDP):
            model = model.module
        return model

    def _get_matching_module(self, model, module_regex):
        m = self._unwrap(model)
        matched = {n: mod for n, mod in m.named_modules() if re.fullmatch(module_regex, n)}
        if len(matched) != 1:
            raise ValueError(f"Expected 1 module match for {module_regex}, got {len(matched)}")
        return next(iter(matched.values()))

    def _freeze_all_params(self, model, requires_grad=False):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        m = self._unwrap(model)
        for name, param in m.named_parameters():
            if any(re.fullmatch(pat, name) for pat in trainable_params_regex):
                param.requires_grad = requires_grad

    def _get_phase2_params(self):
        m = self._unwrap(self.model)
        return [
            p for n, p in m.named_parameters()
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

    # ------------------------------------------------------------------ #
    #  Two-phase orchestration                                            #
    # ------------------------------------------------------------------ #

    def evaluate(self, *args, **kwargs):
        if self._phase == 1:
            return {}
        return super().evaluate(*args, **kwargs)

    def train(self, resume_from_checkpoint=None, **kwargs):
        original_epochs = self.args.num_train_epochs

        # With most params frozen + gradient_checkpointing=True, embedding output
        # has requires_grad=False so checkpointed layers see no grad-bearing inputs
        # and the loss loses its grad_fn. HF only auto-installs this hook when PEFT
        # is loaded (modeling_utils.py:2421); install it ourselves.
        # Also force use_reentrant=False: phase 1 calls torch.autograd.grad with
        # create_graph=True for the conflict-cosine, and reentrant checkpointing
        # silently returns zero gradients under double-backward.
        if getattr(self.args, "gradient_checkpointing", False):
            self._unwrap(self.model).enable_input_require_grads()
            gc_kwargs = self.args.gradient_checkpointing_kwargs or {}
            if gc_kwargs.get("use_reentrant", True):
                gc_kwargs = {**gc_kwargs, "use_reentrant": False}
                self.args.gradient_checkpointing_kwargs = gc_kwargs

        # Phase 1 — skipped entirely when encoder_epochs <= 0 so phase 2 can be
        # tested in isolation. The encoder stays at its init (alpha-residual MLP
        # → r ≈ scaled input direction once normalized).
        if self.encoder_epochs > 0:
            self._phase = 1
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            self.args.num_train_epochs = self.encoder_epochs
            super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
            resume_from_checkpoint = None

        # Phase 2: freeze encoder, unlearn LLM
        self._phase = 2
        self._freeze_all_params(self._encoder_module(), requires_grad=False)
        self._encoder_module().eval()
        self.optimizer = None
        self.lr_scheduler = None
        self.args.num_train_epochs = original_epochs - self.encoder_epochs
        super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        self.args.num_train_epochs = original_epochs

    def create_optimizer(self):
        # Wrap encoder in DDP on first call (dist is initialized by now)
        self._maybe_wrap_encoder()

        if self._phase == 1:
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            optimizer_cls, _ = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(
                self._encoder_raw.parameters(),
                lr=self.encoder_lr,
            )
        else:
            self._freeze_all_params(self._encoder_module(), requires_grad=False)
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            super().create_optimizer()

    # ------------------------------------------------------------------ #
    #  Steering vector                                                    #
    # ------------------------------------------------------------------ #

    def _ramped_steering_coeff(self) -> float:
        if self.coeff_warmup_steps <= 0:
            return self.steering_coeff
        return self.steering_coeff * max(0.0, 1.0 - self._phase2_step / self.coeff_warmup_steps)

    def _compute_steering(self, h_forget: torch.Tensor, steering_coeff: float):
        """(r_normed, r_scaled_unsqueezed)."""
        pooled = h_forget.float().mean(dim=1)
        r = self.encoder(pooled)
        r_normed = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
        r_scaled = r_normed.unsqueeze(1).to(h_forget.dtype) * steering_coeff
        return r_normed, r_scaled

    def _build_steering_target(self, h_forget_ref: torch.Tensor, r_scaled: torch.Tensor) -> torch.Tensor:
        return h_forget_ref + r_scaled

    # ------------------------------------------------------------------ #
    #  Loss computation                                                   #
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
        phase2_params = self._get_phase2_params()

        # Compute retain grad FIRST and detach, so the retain forward graph is
        # released before we hold the forget graph + recomputed activations from
        # autograd.grad(create_graph=True). Otherwise peak memory holds both
        # forwards plus the recompute and we OOM on Llama-2-7b under non-reentrant
        # gradient checkpointing.
        retain_loss_for_conflict = self.model(**retain_inputs).loss
        grad_retain = torch.autograd.grad(
            retain_loss_for_conflict, phase2_params,
            create_graph=False, allow_unused=True,
        )
        grad_retain = [g.detach() if g is not None else None for g in grad_retain]
        retain_nll_value = retain_loss_for_conflict.detach().clone()
        del retain_loss_for_conflict

        h_forget, _ = self.forward_with_cache(
            self.model, forget_inputs, self.model_module, no_grad=False
        )
        with torch.no_grad():
            h_forget_ref, _ = self.forward_with_cache(
                self.ref_model, forget_inputs, self.ref_module, no_grad=True
            )
        h_forget_ref = h_forget_ref.to(h_forget.dtype)

        r_normed, r_scaled = self._compute_steering(h_forget, self.steering_coeff)
        target = self._build_steering_target(h_forget_ref, r_scaled)

        forget_loss = self.compute_activation_loss(
            h_forget.float(), target.float(), mask
        )

        # Per-rank gradient computation. Note that torch.autograd.grad
        # bypasses DDP's gradient sync hooks, so g1/g2 are LOCAL to this
        # rank. Conflict cosine is therefore a per-rank estimate. The
        # encoder's gradient w.r.t. these still propagates correctly via
        # standard backward+DDP-sync at the end.
        grad_forget = torch.autograd.grad(
            forget_loss, phase2_params,
            create_graph=True, allow_unused=True, retain_graph=True,
        )

        paired = [(a, b) for a, b in zip(grad_forget, grad_retain)
                  if a is not None and b is not None]

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

        # Logged values: average across ranks for cleaner monitoring
        with torch.no_grad():
            displacement = (target - h_forget).norm(dim=-1).mean()
            log_tensors = {
                "phase1/conflict_cos": torch.tensor(cos_sim_val, device=h_forget.device),
                "phase1/conflict_cos2": grad_conflict.detach().clone(),
                "phase1/anchor_loss": anchor_loss.detach().clone(),
                "phase1/forget_loss_sim": forget_loss.detach().clone(),
                "phase1/retain_nll": retain_nll_value,
                "phase1/displacement_h_to_target": displacement,
                "phase1/g_forget_norm": torch.tensor(gf_norm, device=h_forget.device),
                "phase1/g_retain_norm": torch.tensor(gr_norm, device=h_forget.device),
            }
            for v in log_tensors.values():
                _all_reduce_mean_(v)
            log_dict = {k: v.item() for k, v in log_tensors.items()}

        if _rank() == 0:
            self.log(log_dict)
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

        with torch.no_grad():
            log_tensors = {
                "train/forget_loss": forget_loss.detach().clone(),
                "train/retain_loss": retain_loss.detach().clone(),
                "train/displacement": r_scaled.norm(dim=-1).mean(),
            }
            for v in log_tensors.values():
                _all_reduce_mean_(v)
            log_dict = {k: v.item() for k, v in log_tensors.items()}
            log_dict["train/forget_warmup_coeff"] = warmup_coeff
            log_dict["train/steering_coeff"] = self._ramped_steering_coeff()

        if _rank() == 0:
            self.log(log_dict)
        return loss, forget_outputs

    # ------------------------------------------------------------------ #
    #  Retain loss                                                        #
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