import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from trainer.unlearn.grad_diff import GradDiff


class PerSampleEncoder(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int = 256, num_layers: int = 2):
        """MLP encoder: hidden_size → latent_dim → [latent_dim →]* hidden_size.

        num_layers controls depth (minimum 2):
          2 = Linear-GELU-Linear  (original)
          3 = Linear-GELU-Linear-GELU-Linear
          etc.
        """
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        layers = [nn.Linear(hidden_size, latent_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(latent_dim, latent_dim), nn.GELU()]
        layers.append(nn.Linear(latent_dim, hidden_size))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class LatentRMU(GradDiff):
    def __init__(
        self,
        module_regex="model\.layers\.7",
        trainable_params_regex=["model\.layers\.(5|6|7|8)\.mlp\..*\.weight"],
        steering_coeff=20,
        latent_dim=256,
        encoder_epochs=2,
        encoder_layers=2,
        orth_weight=1.0,
        retain_sep_weight=1.0,
        encoder_lr=1e-3,
        retain_pca_components=64,
        forget_warmup_steps=0,
        anchor_weight=1.0,
        retain_basis_batches=20,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        self.trainable_params_regex = trainable_params_regex
        self.module_regex = module_regex
        self.model_module = self._get_matching_module(self.model, self.module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, self.module_regex)
        self.steering_coeff = steering_coeff
        self.encoder_epochs = encoder_epochs
        self.orth_weight = orth_weight
        self.retain_sep_weight = retain_sep_weight
        self.encoder_lr = encoder_lr
        self.forget_warmup_steps = forget_warmup_steps
        self.anchor_weight = anchor_weight
        self.retain_basis_batches = retain_basis_batches
        self._phase = 1
        self._phase2_step = 0

        hidden_size = self.model.config.hidden_size
        self.encoder = PerSampleEncoder(hidden_size, latent_dim)
        self.encoder.to(next(self.model.parameters()).device)

        self.retain_pca_components = 256

        # after self._phase2_step = 0
        self._retain_basis = None       # (k, H) float32, rows are orthonormal retain directions
        self._retain_mean = None        # (H,) float32, for centering
        self._basis_step = 0            # counts Phase 1 steps since last refresh

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_dontknow_activations(self, forget_inputs):
        input_ids = forget_inputs["input_ids"].clone()
        labels = forget_inputs["labels"]

        # Mask answer tokens with [UNK] or a pad token — model sees question
        # structure but answer positions are blanked
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
        return F.normalize(h.float().mean(dim=1), dim=-1)  # (B, H)
    
    def _get_matching_module(self, model, module_regex):
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module
        matched = {n: m for n, m in model.named_modules() if re.fullmatch(module_regex, n)}
        if len(matched) == 0:
            raise ValueError(f"No module matches for {module_regex}")
        return list(matched.values())

    def _freeze_all_params(self, model, requires_grad=False):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        for name, param in model.named_parameters():
            if any(re.fullmatch(pat, name) for pat in trainable_params_regex):
                param.requires_grad = requires_grad

    def _get_phase2_params(self):
        """Parameters Phase 2 will train — used for gradient conflict in Phase 1."""
        return [
            p for n, p in self.model.named_parameters()
            if any(re.fullmatch(pat, n) for pat in self.trainable_params_regex)
            and p.requires_grad
        ]

    def _build_retain_basis(self):
        """Collect retain activations from ref_model and compute top-k PCA basis."""
        self.ref_model.eval()
        acts = []
        # Grab an iterator over retain data. Trainer builds train_dataloader from the
        # combined forget+retain dataset; we pull retain_inputs out of each batch.
        loader = self.get_train_dataloader()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= self.retain_basis_batches:
                    break
                retain_inputs = {
                    k: batch["retain"][k].to(device)
                    for k in ("input_ids", "attention_mask", "labels")
                }
                h, _ = self.forward_with_cache(
                    self.ref_model, retain_inputs, self.ref_module, no_grad=True
                )  # (B, S, H)
                mask = (retain_inputs["labels"] != -100).unsqueeze(-1).float()  # (B,S,1)
                pooled = (h.float() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, H)
                acts.append(pooled.cpu())

        X = torch.cat(acts, dim=0)                      # (N, H)
        mean = X.mean(dim=0, keepdim=True)              # (1, H)
        Xc = X - mean
        k = min(self.retain_pca_components, Xc.shape[0] - 1, Xc.shape[1])
        _, _, V = torch.pca_lowrank(Xc, q=k, center=False)  # V: (H, k)
        basis = V.T.contiguous().to(device).float()     # (k, H)

        self._retain_basis = basis
        self._retain_mean = mean.squeeze(0).to(device).float()
        self._basis_step = 0
        print(f"[DEBUG] built retain basis: {basis.shape}, "
            f"explained_var_ratio≈{(Xc @ V).pow(2).sum() / Xc.pow(2).sum():.4f}", flush=True)


    def _subspace_conflict(self, r: torch.Tensor) -> torch.Tensor:
        """Fraction of squared energy of r that lies in the retain subspace.

        r: (B, H) — encoder output, before/after norm rescaling both work since
        we normalize here explicitly. Returns a scalar in [0, 1].
        """
        r_centered = r - self._retain_mean.to(r.dtype)
        r_unit = F.normalize(r_centered, dim=-1)           # (B, H)
        basis = self._retain_basis.to(r.dtype)             # (k, H)
        proj = r_unit @ basis.T                            # (B, k)
        # Squared cosine energy captured by retain subspace, averaged over batch.
        return proj.pow(2).sum(dim=-1).mean()

    def forward_with_cache(self, model, inputs, module, no_grad=True):
        modules = module if isinstance(module, list) else [module]
        caches = {id(m): [] for m in modules}
        def make_hook(mid):
            def hook(m, inp, out):
                caches[mid].append(out[0] if isinstance(out, tuple) else out)
            return hook
        handles = [m.register_forward_hook(make_hook(id(m))) for m in modules]
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
        for h in handles:
            h.remove()
        acts = [caches[id(m)][0] for m in modules]
        h_avg = torch.stack(acts, dim=0).mean(dim=0)
        return h_avg, outputs

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

        # Phase 1: train encoder only, but keep phase2 params unfrozen
        # so gradient conflict can be computed through them
        self._phase = 1
        self._freeze_all_params(self.model, requires_grad=False)
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        self.args.num_train_epochs = self.encoder_epochs
        super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        # Phase 2: freeze encoder, unlearn LLM
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
            # Phase 2 params have requires_grad=True for conflict computation,
            # but are NOT in the optimizer — only encoder params are updated
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
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
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def _cosine_conflict(self, grad1, grad2):
        """Cosine conflict between two gradient lists.

        Returns cos_sim^2 — zero only when gradients are exactly orthogonal, positive otherwise.
        Minimizing this always provides a gradient signal pushing toward orthogonality,
        regardless of whether gradients are aligned or conflicting.
        """
        g1_list = [g.flatten() for g in grad1 if g is not None]
        g2_list = [g.flatten() for g in grad2 if g is not None]
        print(f"[DEBUG] grad1 nones={sum(g is None for g in grad1)}/{len(grad1)}, grad2 nones={sum(g is None for g in grad2)}/{len(grad2)}", flush=True)
        if not g1_list or not g2_list:
            print(f"[DEBUG] empty grad lists!", flush=True)
            return torch.tensor(0.0, requires_grad=True)
        g1 = torch.cat(g1_list)
        g2 = torch.cat(g2_list)
        cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).squeeze()
        print(f"[DEBUG] g1 norm={g1.norm().item():.4f}, g2 norm={g2.norm().item():.4f}, cos_sim={cos_sim.item():.4f}, result={( cos_sim**2).item():.6f}", flush=True)
        return cos_sim ** 2

    def _get_steering_vectors(self, h_forget: torch.Tensor) -> torch.Tensor:
        pooled = h_forget.float().mean(dim=1)
        r = self.encoder(pooled)
        r = r / (r.norm(dim=-1, keepdim=True) + 1e-8) * self.steering_coeff
        return r.unsqueeze(1).to(h_forget.dtype)

    def _encode_steering(self, h_forget: torch.Tensor) -> torch.Tensor:
        pooled = h_forget.float().mean(dim=1)
        r = self.encoder(pooled)
        r = r / (r.norm(dim=-1, keepdim=True) + 1e-8) * self.steering_coeff
        return r.unsqueeze(1).expand_as(h_forget)

    def _create_intervention_hook(self, steering_vector: torch.Tensor, coeff: float = None):
        """Hook that adds a steering displacement relative to the activation norm.

        displacement = direction * ||activation|| * coeff

        Args:
            steering_vector: (hidden,) for a single shared vector, or
                             (batch, hidden) for per-sample vectors.
            coeff: scaling coefficient. Defaults to self.steering_coeff.
        """
        if coeff is None:
            coeff = self.steering_coeff

        if steering_vector.dim() == 1:
            direction = steering_vector.unsqueeze(0).unsqueeze(0)  # (1,1,H)
        else:
            direction = steering_vector.unsqueeze(1)  # (B,1,H)
        direction = direction / (direction.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        def intervention_hook(module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            scale = h.norm(p=2, dim=-1, keepdim=True) * coeff  # (B,S,1)
            modified = h + direction * scale
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return intervention_hook

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = {k: inputs["forget"][k] for k in ("input_ids", "attention_mask", "labels")}
        retain_inputs = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        mask = forget_inputs["labels"] != -100

        # if self._phase == 1:
        #     h_forget, _ = self.forward_with_cache(
        #         self.model, forget_inputs, self.model_module, no_grad=False
        #     )

        #     pooled = h_forget.float().mean(dim=1)
        #     r = self.encoder(pooled)
        #     r = r / (r.norm(dim=-1, keepdim=True) + 1e-8) * self.steering_coeff
        #     r_expanded = r.unsqueeze(1).expand_as(h_forget)

        #     forget_loss = self.compute_activation_loss(
        #         h_forget.float(), r_expanded.float(), mask
        #     )

        #     retain_loss = self.compute_retain_loss(self.model, retain_inputs)

        #     phase2_params = self._get_phase2_params()
        #     print(f"[DEBUG] n_phase2_params={len(phase2_params)}", flush=True)

        #     grad_forget = torch.autograd.grad(
        #         forget_loss, phase2_params,
        #         create_graph=True, allow_unused=True, retain_graph=True,
        #     )

        #     grad_retain = torch.autograd.grad(
        #         retain_loss, phase2_params,
        #         create_graph=False, allow_unused=True,
        #     )
        #     grad_retain = [g.detach() if g is not None else None for g in grad_retain]

        #     # FIX 2: only compute conflict over params where BOTH gradients are defined
        #     paired = [
        #         (g1, g2) for g1, g2 in zip(grad_forget, grad_retain)
        #         if g1 is not None and g2 is not None
        #     ]
        #     print(f"[DEBUG] forget_loss={forget_loss.item():.4f}, retain_loss={retain_loss.item():.4f}, "
        #         f"shared_params={len(paired)}/{len(phase2_params)}", flush=True)

        #     # Retain separation: penalize r for being aligned with retain activations.
        #     # This directly trains the encoder to produce forget-specific directions
        #     # that are orthogonal to the retain subspace, complementing gradient conflict.
        #     h_retain_ref, _ = self.forward_with_cache(
        #         self.ref_model, retain_inputs, self.ref_module, no_grad=True
        #     )
        #     retain_pooled = h_retain_ref.float().mean(dim=1)  # (B, H)
        #     r_unit = F.normalize(r, dim=-1)                    # (B, H)
        #     retain_unit = F.normalize(retain_pooled, dim=-1)   # (B, H)
        #     retain_sep_loss = (r_unit * retain_unit).sum(dim=-1).abs().mean()

        #     if not paired:
        #         print("[DEBUG] no shared params with grad — check module_regex covers trainable layers", flush=True)
        #         grad_conflict_term = torch.tensor(0.0, device=h_forget.device)
        #     else:
        #         g1 = torch.cat([g1.flatten() for g1, _ in paired])
        #         g2 = torch.cat([g2.flatten() for _, g2 in paired])
        #         cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).squeeze()
        #         grad_conflict_term = cos_sim ** 2
        #         print(f"[DEBUG] g1 norm={g1.norm().item():.4f}, g2 norm={g2.norm().item():.4f}, "
        #             f"cos_sim={cos_sim.item():.4f}, conflict={grad_conflict_term.item():.6f}, "
        #             f"retain_sep={retain_sep_loss.item():.4f}", flush=True)

        #     loss = self.orth_weight * grad_conflict_term + self.retain_sep_weight * retain_sep_loss

        ########## OLD CODE FROM HERE ##########
        # if self._phase == 1:
        #     h_forget, _ = self.forward_with_cache(
        #         self.model, forget_inputs, self.model_module, no_grad=False
        #     )

        #     r_expanded = h_forget + self._encode_steering(h_forget)

        #     forget_loss = self.compute_activation_loss(
        #         h_forget.float(), r_expanded.float(), mask
        #     )

        #     # For grad_retain we need a loss that is non-zero even when model == ref.
        #     # EMBED_DIFF = MSE(model_act, ref_act) = 0 throughout Phase 1 because
        #     # phase2_params are frozen, so the model never diverges from ref.
        #     # Use NLL instead, which has a non-zero gradient at any model state.
        #     retain_loss_for_conflict = self.model(**retain_inputs).loss

        #     phase2_params = self._get_phase2_params()
        #     print(f"[DEBUG] n_phase2_params={len(phase2_params)}", flush=True)

        #     if phase2_params and self.orth_weight > 0.0:
        #         grad_forget = torch.autograd.grad(
        #             forget_loss, phase2_params,
        #             create_graph=True, allow_unused=True, retain_graph=True,
        #         )

        #         grad_retain = torch.autograd.grad(
        #             retain_loss_for_conflict, phase2_params,
        #             create_graph=False, allow_unused=True,
        #         )
        #         grad_retain = [g.detach() if g is not None else None for g in grad_retain]

        #         paired = [
        #             (g1, g2) for g1, g2 in zip(grad_forget, grad_retain)
        #             if g1 is not None and g2 is not None
        #         ]
        #     else:
        #         paired = []
        #     print(f"[DEBUG] forget_loss={forget_loss.item():.4f}, retain_nll={retain_loss_for_conflict.item():.4f}, "
        #         f"shared_params={len(paired)}/{len(phase2_params)}", flush=True)

        #     h_retain_ref, _ = self.forward_with_cache(
        #         self.ref_model, retain_inputs, self.ref_module, no_grad=True
        #     )
        #     retain_pooled = h_retain_ref.float().mean(dim=1)  # (B, H)
        #     r_unit = F.normalize(r_expanded[:, 0, :].float(), dim=-1)   # (B, H)
        #     retain_unit = F.normalize(retain_pooled, dim=-1)   # (B, H)
        #     retain_sep_loss = (r_unit * retain_unit).sum(dim=-1).abs().mean()

        #     # anchor: pull encoder direction toward "can't answer" activations
        #     dontknow_dirs = self._get_dontknow_activations(forget_inputs)  # (B, H)
        #     anchor_loss = 1 - (r_unit * dontknow_dirs).sum(dim=-1).mean()

        #     if not paired:
        #         print("[DEBUG] no shared params with grad — check module_regex covers trainable layers", flush=True)
        #         grad_conflict_term = torch.tensor(0.0, device=h_forget.device)
        #     else:
        #         g1 = torch.cat([g1.flatten() for g1, _ in paired])
        #         g2 = torch.cat([g2.flatten() for _, g2 in paired])
        #         cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).squeeze()
        #         grad_conflict_term = cos_sim ** 2
        #         print(f"[DEBUG] g1 norm={g1.norm().item():.4f}, g2 norm={g2.norm().item():.4f}, "
        #             f"cos_sim={cos_sim.item():.4f}, conflict={grad_conflict_term.item():.6f}, "
        #             f"retain_sep={retain_sep_loss.item():.4f}, anchor={anchor_loss.item():.4f}", flush=True)

        #     loss = (self.orth_weight * grad_conflict_term
        #         + self.retain_sep_weight * retain_sep_loss
        #         + self.anchor_weight * anchor_loss)

        if self._phase == 1:
            # Refresh retain basis if needed.
            if self._retain_basis is None or self._basis_step >= self.retain_basis_batches:
                self._build_retain_basis()
            self._basis_step += 1

            h_forget, _ = self.forward_with_cache(
                self.model, forget_inputs, self.model_module, no_grad=False
            )

            # # Encoder forward
            # pooled = h_forget.float().mean(dim=1)                      # (B, H)
            # r = self.encoder(pooled)                                   # (B, H)
            # r_scaled = r / (r.norm(dim=-1, keepdim=True) + 1e-8) * self.steering_coeff
            # r_expanded = (h_forget + r_scaled.unsqueeze(1)).to(h_forget.dtype)
            r_expanded = self._encode_steering(h_forget)
            r = r_expanded[:, 0, :]  # (B, H) — the steering vector added to the first token's activation, which is what we use for conflict and subspace losses
            # Forget loss: MSE against steered target (same as before)
            forget_loss = self.compute_activation_loss(
                h_forget.float(), r_expanded.float(), mask
            )

            # Subspace conflict: how much of r lives in the retain PCA subspace?
            subspace_term = self._subspace_conflict(r)  # (B, H) → (B,)

            # Anchor toward don't-know directions (unchanged)
            dontknow_dirs = self._get_dontknow_activations(forget_inputs)  # (B, H)
            r_unit_for_anchor = F.normalize(r.float(), dim=-1)
            anchor_loss = 1 - (r_unit_for_anchor * dontknow_dirs).sum(dim=-1).mean()

            h_retain_ref, _ = self.forward_with_cache(
                self.ref_model, retain_inputs, self.ref_module, no_grad=True
            )
            retain_pooled = h_retain_ref.float().mean(dim=1)  # (B, H)
            r_unit = F.normalize(r_expanded[:, 0, :].float(), dim=-1)   # (B, H)
            retain_unit = F.normalize(retain_pooled, dim=-1)   # (B, H)
            retain_sep_loss = (r_unit * retain_unit).sum(dim=-1).abs().mean()

            print(f"[DEBUG] forget_loss={forget_loss.item():.4f}, "
                f"subspace={subspace_term.item():.4f}, "
                f"anchor={anchor_loss.item():.4f}, "
                f"retain_sep={retain_sep_loss.item():.4f}", flush=True)

            loss = (self.orth_weight * subspace_term
                    + self.anchor_weight * anchor_loss
                    + self.retain_sep_weight * retain_sep_loss)
            # Note: forget_loss is not added. The encoder's job in Phase 1 is to find
            # a direction that (a) is outside the retain subspace and (b) points toward
            # don't-know activations. The MSE-against-steered-target signal belongs to
            # Phase 2. Include `forget_loss` here with a small weight if you want the
            # encoder to also care about magnitude realizability.
        else:
            h_forget, forget_outputs = self.forward_with_cache(
                self.model, forget_inputs, self.model_module, no_grad=False
            )

            with torch.no_grad():
                # calc h_forget_ref
                h_forget_ref, _ = self.forward_with_cache(
                    self.ref_model, forget_inputs, self.ref_module, no_grad=True
                )
                control_vec = self._get_steering_vectors(h_forget_ref)
            control_vec = control_vec.expand_as(h_forget)
            forget_loss = self.compute_activation_loss(h_forget, control_vec, mask)
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
            })

        return (loss, forget_outputs) if (return_outputs and self._phase == 2) else loss
    
    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "EMBED_DIFF":
            model_retain_act, _ = self.forward_with_cache(
                model, retain_inputs, self.model_module, no_grad=False
            )
            ref_retain_act, _ = self.forward_with_cache(
                self.ref_model, retain_inputs, self.ref_module, no_grad=True
            )
            mask = retain_inputs["labels"] != -100
            return self.compute_activation_loss(
                model_retain_act, ref_retain_act.to(model_retain_act.device), mask,
            )
        return super().compute_retain_loss(model, retain_inputs)