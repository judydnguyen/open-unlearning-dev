from typing import Any, Dict, List, Optional, Tuple, Union
import contextlib
import copy
import functools
import logging
import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Any]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Any]],
    **kwargs,
):
    """
    Context manager for temporarily adding forward hooks to a model.
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for activation summarization.
    Uses a learnable query to attend over sequence positions.
    """
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, h: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            pooled: (batch_size, hidden_size)
        """
        batch_size, seq_len, hidden_size = h.shape

        # Project keys and values
        keys = self.key_proj(h)  # (B, S, H)
        values = self.value_proj(h)  # (B, S, H)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, H)

        # Reshape for multi-head attention
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, 1, head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, S, head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, S, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale  # (B, heads, 1, S)

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, 1, S)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, values)  # (B, heads, 1, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, hidden_size)  # (B, 1, H)

        # Final projection and squeeze
        pooled = self.out_proj(attn_output).squeeze(1)  # (B, H)

        return pooled


class PerSampleEncoder(nn.Module):
    """
    Per-sample encoder for forget set activations.
    Each forget sample gets its own latent representation: z_i = ρ(ψ(h_i)).
    No mean-pooling — every sample produces an independent steering vector.
    """
    def __init__(self, hidden_size: int, latent_dim: int = 256):
        super().__init__()
        # ψ: per-sample feature extractor
        self.psi = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # ρ: per-sample refinement
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch_size, hidden_size) pooled activations
        Returns:
            z: (batch_size, latent_dim) per-sample encodings
        """
        psi_h = self.psi(h)          # (batch_size, latent_dim)
        z = self.rho(psi_h)          # (batch_size, latent_dim)
        return z


class MappingNetwork(nn.Module):
    """
    Mapping network g_φ that produces per-sample steering vectors.
    Implements: r_ℓ,i = c · tanh(W_φ z_i) for each sample i.

    For multi-layer support, outputs steering vectors for multiple layers.
    """
    def __init__(self, latent_dim: int, hidden_size: int, c: float = 1.0, num_layers: int = 1):
        super().__init__()
        self.c = c
        self.num_layers = num_layers

        if num_layers == 1:
            self.W = nn.Linear(latent_dim, hidden_size)
        else:
            # Separate projection for each layer
            self.W = nn.ModuleList([
                nn.Linear(latent_dim, hidden_size) for _ in range(num_layers)
            ])
            # Learnable per-layer scaling
            self.layer_scales = nn.Parameter(torch.ones(num_layers))

    def forward(self, z_f: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            z_f: (batch_size, latent_dim) per-sample encodings
        Returns:
            r_ell: (batch_size, hidden_size) or list of (batch_size, hidden_size) per-sample steering vectors
        """
        if self.num_layers == 1:
            r_ell = self.c * torch.tanh(self.W(z_f))
            return r_ell
        else:
            r_ells = []
            for i, w in enumerate(self.W):
                scale = self.layer_scales[i]
                r_ell = self.c * scale * torch.tanh(w(z_f))
                r_ells.append(r_ell)
            return r_ells


class OnlinePCAEstimator(nn.Module):
    """
    Online PCA estimator using incremental SVD for retain set orthogonality.
    Maintains a running estimate of the principal components of retain activations.
    """
    def __init__(self, hidden_size: int, n_components: int = 64, momentum: float = 0.99):
        super().__init__()
        self.n_components = n_components
        self.momentum = momentum

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(hidden_size))
        self.register_buffer('running_cov', torch.eye(hidden_size) * 0.01)
        self.register_buffer('n_samples', torch.tensor(0, dtype=torch.long))
        self.register_buffer('pca_basis', torch.zeros(hidden_size, n_components))

    @torch.no_grad()
    def update(self, h: torch.Tensor):
        """
        Update running statistics with new activations.
        Args:
            h: (batch_size, hidden_size) activations
        """
        batch_size = h.shape[0]

        # Update running mean
        batch_mean = h.mean(dim=0)
        if self.n_samples == 0:
            self.running_mean.copy_(batch_mean)
        else:
            self.running_mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)

        # Update running covariance
        h_centered = h - self.running_mean
        batch_cov = torch.matmul(h_centered.T, h_centered) / batch_size
        if self.n_samples == 0:
            self.running_cov.copy_(batch_cov)
        else:
            self.running_cov.mul_(self.momentum).add_(batch_cov, alpha=1 - self.momentum)

        self.n_samples.add_(batch_size)

        # Periodically update PCA basis
        if self.n_samples % 100 < batch_size:
            self._update_pca_basis()

    @torch.no_grad()
    def _update_pca_basis(self):
        """Compute PCA basis from running covariance."""
        try:
            # Use eigendecomposition for PCA
            eigenvalues, eigenvectors = torch.linalg.eigh(self.running_cov)
            # Take top n_components (sorted ascending, so take from end)
            self.pca_basis.copy_(eigenvectors[:, -self.n_components:])
        except Exception as e:
            logger.warning(f"PCA update failed: {e}")

    def get_basis(self) -> Optional[torch.Tensor]:
        """Get current PCA basis if enough samples collected."""
        if self.n_samples < 100:
            return None
        return self.pca_basis


class LatentUnlearning(UnlearnTrainer):
    """
    Latent Unlearning trainer with two-phase training.

    Phase 1 (Encoder Training):
      Trains the DeepSets encoder + mapping network g_φ to produce steering
      vectors r_ℓ that suppress forget-set likelihood while preserving utility.
      The LLM is frozen; only encoder parameters are optimized.

    Phase 2 (LLM Unlearning — RMU-style):
      Freezes the trained encoder and uses its steering vectors as misdirection
      targets (replacing RMU's random control vector). The LLM is unfrozen and
      trained with activation-level MSE: forget activations are pushed toward
      the encoder target, retain activations are anchored to a frozen reference.

    Features:
    - Multi-layer intervention support
    - Attention-based activation pooling
    - Online PCA estimation for orthogonality constraint
    """

    def __init__(
        self,
        intervention_layer: Optional[int] = None,
        intervention_layers: Optional[List[int]] = None,
        lambda_util: float = 1.0,
        mu_orth: float = 1.0,
        rho_norm: float = 1.0,
        c: float = 1.0,
        latent_dim: int = 256,
        retain_pca_basis: Optional[torch.Tensor] = None,
        kl_last_token_only: bool = True,
        pooling_type: str = "mean",  # "mean" or "attention"
        use_online_pca: bool = False,
        online_pca_components: int = 64,
        encoder_epochs: int = 2,
        phase2_learning_rate: Optional[float] = None,
        steering_coeff: float = 20.0,
        intervention_coeff: float = 0.1,
        forget_warmup_steps: int = 0,
        forget_loss_type: str = "mse",
        entropy_weight: float = 1.0,
        max_steering_norm: float = 5.0,
        centroid_noise: float = 0.0,
        repulsion_weight: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Handle single layer or multi-layer intervention
        if intervention_layers is not None:
            self.intervention_layers = intervention_layers
        elif intervention_layer is not None:
            self.intervention_layers = [intervention_layer]
        else:
            raise ValueError("Either intervention_layer or intervention_layers must be specified")

        self.lambda_util = lambda_util
        self.mu_orth = mu_orth
        self.rho_norm = rho_norm
        self.c = c
        self.kl_last_token_only = kl_last_token_only
        self.pooling_type = pooling_type
        self.use_online_pca = use_online_pca
        self.encoder_epochs = encoder_epochs
        self.phase2_learning_rate = phase2_learning_rate
        self.steering_coeff = steering_coeff
        self.intervention_coeff = intervention_coeff
        self.forget_warmup_steps = forget_warmup_steps
        self.forget_loss_type = forget_loss_type
        self.entropy_weight = entropy_weight
        self.max_steering_norm = max_steering_norm
        self.centroid_noise = centroid_noise
        self.repulsion_weight = repulsion_weight
        self._phase = 1
        self.ref_model = None

        # Get hidden size from model
        hidden_size = self._get_hidden_size()
        if hidden_size is None:
            raise ValueError("Could not determine hidden_size from model config")
        self.hidden_size = hidden_size

        # Initialize per-sample encoder and mapping network
        self.encoder = PerSampleEncoder(hidden_size, latent_dim)
        self.mapping_network = MappingNetwork(
            latent_dim, hidden_size, c, num_layers=len(self.intervention_layers)
        )

        # Attention pooling if requested
        if pooling_type == "attention":
            self.attention_pooling = AttentionPooling(hidden_size)
        else:
            self.attention_pooling = None

        # Move mapping networks to same device as model
        device = next(self.model.parameters()).device
        self.encoder.to(device)
        self.mapping_network.to(device)
        if self.attention_pooling is not None:
            self.attention_pooling.to(device)

        # Register as submodules BEFORE DDP wrapping (in __init__, not create_optimizer)
        self._latent_modules = nn.ModuleDict({
            "encoder": self.encoder,
            "mapping_network": self.mapping_network,
        })
        if self.attention_pooling is not None:
            self._latent_modules["attention_pooling"] = self.attention_pooling

        # Online PCA estimator
        if use_online_pca:
            self.online_pca = OnlinePCAEstimator(hidden_size, online_pca_components)
            self.online_pca.to(device)
            self._latent_modules["online_pca"] = self.online_pca
        else:
            self.online_pca = None

        # Store retain PCA basis (can be pre-computed or from online estimation)
        if retain_pca_basis is not None:
            self.register_buffer('retain_pca_basis', retain_pca_basis.to(device))
        else:
            self.retain_pca_basis = None

        # Freeze the main model
        for param in self.model.parameters():
            param.requires_grad = False

        # For frozen LLMs, default to eval to avoid dropout noise
        self.model.eval()

        # Ensure mapping networks are in training mode
        self.encoder.train()
        self.mapping_network.train()
        if self.attention_pooling is not None:
            self.attention_pooling.train()

    def _get_hidden_size(self) -> Optional[int]:
        """Get hidden size from model config."""
        model = self._unwrap_model(self.model)
        if hasattr(model, "config"):
            return getattr(model.config, "hidden_size", None)
        return None

    def _unwrap_model(self, model):
        """Unwrap DDP/DeepSpeed/FSDP wrappers."""
        unwrapped = model

        # Handle DeepSpeed engine
        try:
            import deepspeed
            if isinstance(unwrapped, deepspeed.DeepSpeedEngine):
                unwrapped = unwrapped.module
        except ImportError:
            pass

        # Handle DDP/FSDP wrappers
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        return unwrapped

    def _get_layer_module(self, model, layer_idx: int):
        """Get the module at the specified layer index."""
        unwrapped = self._unwrap_model(model)

        # Find layers in common model architectures
        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "layers"):
            # LlamaForCausalLM, MistralForCausalLM, etc.
            return unwrapped.model.layers[layer_idx]
        elif hasattr(unwrapped, "transformer") and hasattr(unwrapped.transformer, "h"):
            # GPT-2, GPT-Neo, etc.
            return unwrapped.transformer.h[layer_idx]
        elif hasattr(unwrapped, "layers"):
            return unwrapped.layers[layer_idx]
        else:
            raise ValueError(f"Could not find layers in model at index {layer_idx}")

    def _precompute_retain_cache(self):
        """Precompute ref model activations for all retain samples.

        Iterates over the retain sub-dataset once, caching each batch's
        inputs and ref-model activations on CPU. During Phase 2 training,
        a random cached batch is sampled each step instead of running
        the ref model on retain data.
        """
        from torch.utils.data import DataLoader

        retain_ds = getattr(self.train_dataset, "retain", None)
        if retain_ds is None:
            raise ValueError("train_dataset has no 'retain' sub-dataset for caching")

        device = next(self.ref_model.parameters()).device
        layer_idx = self.intervention_layers[0]
        batch_size = self.args.per_device_train_batch_size
        retain_loader = DataLoader(
            retain_ds,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        cache = []
        # Accumulators for retain centroid computation
        act_sum = None
        act_sq_sum = None
        total_tokens = 0

        for batch in retain_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ref_act, _ = self._forward_with_cache(
                self.ref_model, batch, layer_idx, no_grad=True
            )
            cache.append({
                "input_ids": batch["input_ids"].cpu(),
                "attention_mask": batch["attention_mask"].cpu(),
                "labels": batch["labels"].cpu(),
                "ref_activations": ref_act.cpu(),
            })

            # Accumulate statistics for centroid (non-padding tokens only)
            mask = (batch["labels"] != -100).unsqueeze(-1).float()  # (B, S, 1)
            ref_act_f = ref_act.float()
            masked_act = ref_act_f * mask  # (B, S, H)
            n_tokens = mask.sum().item()
            if act_sum is None:
                act_sum = masked_act.sum(dim=(0, 1)).cpu()  # (H,)
                act_sq_sum = (masked_act ** 2).sum(dim=(0, 1)).cpu()  # (H,)
            else:
                act_sum += masked_act.sum(dim=(0, 1)).cpu()
                act_sq_sum += (masked_act ** 2).sum(dim=(0, 1)).cpu()
            total_tokens += n_tokens

        # Compute retain centroid and per-dimension std
        self._retain_centroid = act_sum / total_tokens  # (H,)
        variance = act_sq_sum / total_tokens - self._retain_centroid ** 2
        self._retain_std = variance.clamp(min=0).sqrt()  # (H,)
        logger.info(
            "Precomputed %d retain batches for Phase 2 cache "
            "(centroid norm=%.2f, mean std=%.4f, %d tokens)",
            len(cache), self._retain_centroid.norm().item(),
            self._retain_std.mean().item(), int(total_tokens),
        )
        return cache

    def _get_separate_dataloaders(self):
        """Build separate DataLoaders for the forget and retain sub-datasets."""
        from torch.utils.data import DataLoader

        dataset = self.train_dataset
        forget_ds = getattr(dataset, "forget", None)
        retain_ds = getattr(dataset, "retain", None)
        if forget_ds is None or retain_ds is None:
            return None, None

        batch_size = self.args.per_device_train_batch_size
        collator = self.data_collator
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, collate_fn=collator, shuffle=False)
        retain_loader = DataLoader(retain_ds, batch_size=batch_size, collate_fn=collator, shuffle=False)
        return forget_loader, retain_loader

    def _run_plot(self, tag: str):
        """Generate the latent-vector plot if dataloaders are available."""
        forget_loader, retain_loader = self._get_separate_dataloaders()
        if forget_loader is None:
            logger.warning("Cannot plot latent vectors: forget/retain sub-datasets not found")
            return
        output_dir = self.args.output_dir
        path = os.path.join(output_dir, f"latent_vectors_{tag}.png")
        self.plot_latent_vectors(forget_loader, retain_loader, output_path=path)

    @property
    def _needs_encoder(self) -> bool:
        """Whether the current forget_loss_type uses the Phase 1 encoder."""
        return self.forget_loss_type == "mse"

    def train(self, resume_from_checkpoint=None, **kwargs):
        """Two-phase training: encoder training then LLM unlearning."""
        original_epochs = self.args.num_train_epochs
        original_lr = self.args.learning_rate

        # === Phase 1: Train encoder (skip if not needed) ===
        if self._needs_encoder:
            logger.info("=" * 60)
            logger.info("Phase 1: Training encoder independently (%d epochs)", self.encoder_epochs)
            logger.info("=" * 60)
            self._phase = 1
            self._setup_phase1()
            self.args.num_train_epochs = self.encoder_epochs
            super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

            # Plot after Phase 1 (encoder trained, steering vectors ready)
            self._run_plot("after_phase1")
            remaining_epochs = original_epochs - self.encoder_epochs
        else:
            logger.info("=" * 60)
            logger.info("Skipping Phase 1 (encoder not needed for forget_loss_type=%s)", self.forget_loss_type)
            logger.info("=" * 60)
            remaining_epochs = original_epochs

        # === Phase 2: Unlearn with fixed steering targets ===
        if remaining_epochs <= 0:
            logger.warning(
                "No epochs remaining for Phase 2 (total=%d, encoder=%d). "
                "Increase num_train_epochs.", original_epochs, self.encoder_epochs
            )
            self.args.num_train_epochs = original_epochs
            self.args.learning_rate = original_lr
            return

        logger.info("=" * 60)
        logger.info("Phase 2: Unlearning with fixed steering targets (%d epochs)", remaining_epochs)
        logger.info("=" * 60)
        self._phase = 2
        self._setup_phase2()
        self.args.num_train_epochs = remaining_epochs
        if self.phase2_learning_rate is not None:
            self.args.learning_rate = self.phase2_learning_rate

        # Force new optimizer/scheduler for new parameter set
        self.optimizer = None
        self.lr_scheduler = None

        result = super().train(**kwargs)

        # Plot after Phase 2 (model unlearned, see how activations shifted)
        self._run_plot("after_phase2")

        # Restore original settings
        self.args.num_train_epochs = original_epochs
        self.args.learning_rate = original_lr
        return result

    def _setup_phase1(self):
        """Phase 1: freeze LLM, train encoder."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        for param in self._latent_modules.parameters():
            param.requires_grad = True
        if self.online_pca is not None:
            for param in self.online_pca.parameters():
                param.requires_grad = False

        self.encoder.train()
        self.mapping_network.train()
        if self.attention_pooling is not None:
            self.attention_pooling.train()

    def _setup_phase2(self):
        """Phase 2: freeze encoder, unfreeze LLM, create reference model."""
        # Freeze encoder
        for param in self._latent_modules.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.mapping_network.eval()
        if self.attention_pooling is not None:
            self.attention_pooling.eval()

        # Create frozen reference model for retain activation matching
        logger.info("Creating frozen reference model for Phase 2 retain loss")
        self.ref_model = copy.deepcopy(self.model).to(self.accelerator.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        if self.is_deepspeed_enabled:
            self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare_model(
                self.ref_model, evaluation_mode=True
            )

        # Precompute retain cache before unfreezing model
        self._cached_retain_batches = self._precompute_retain_cache()

        # Unfreeze model
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def get_train_dataloader(self):
        """Phase 1 uses the default interleaved forget/retain loader.
        Phase 2 uses a forget-only loader (retain comes from cache)."""
        if self._phase == 2:
            from torch.utils.data import DataLoader

            forget_ds = getattr(self.train_dataset, "forget", None)
            if forget_ds is None:
                raise ValueError("train_dataset has no 'forget' sub-dataset")

            return self.accelerator.prepare(DataLoader(
                forget_ds,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                shuffle=True,
            ))

        return super().get_train_dataloader()

    def create_optimizer(self):
        """Create optimizer for the current phase's trainable parameters."""
        if self._phase == 1:
            # Phase 1: only optimize encoder parameters
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self._latent_modules.parameters():
                param.requires_grad = True
            if self.online_pca is not None:
                for param in self.online_pca.parameters():
                    param.requires_grad = False
        else:
            # Phase 2: only optimize LLM parameters
            for param in self._latent_modules.parameters():
                param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = True

        super().create_optimizer()

    def _pool_activations(self, h: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool layer activations using mean or attention pooling.
        Args:
            h: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            pooled: (batch_size, hidden_size)
        """
        if self.attention_pooling is not None:
            return self.attention_pooling(h, attention_mask)

        # Default mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            h_masked = h * mask_expanded
            sum_h = h_masked.sum(dim=1)
            seq_lengths = mask_expanded.sum(dim=1)
            pooled = sum_h / (seq_lengths + 1e-8)
        else:
            pooled = h.mean(dim=1)
        return pooled

    def _extract_layer_activations(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Extract and pool activations from specified layer."""
        activations = []

        def activation_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if requires_grad:
                activations.append(output)
            else:
                activations.append(output.detach())

        layer_module = self._get_layer_module(model, layer_idx)
        hook_handle = layer_module.register_forward_hook(activation_hook)
        try:
            if requires_grad:
                _ = model(**{k: v for k, v in inputs.items() if k != "labels"})
            else:
                with torch.no_grad():
                    _ = model(**{k: v for k, v in inputs.items() if k != "labels"})
        finally:
            hook_handle.remove()

        if len(activations) == 0:
            raise RuntimeError("No activations captured")

        h = activations[0]  # (batch, seq, hidden)
        pooled = self._pool_activations(h, inputs.get("attention_mask", None))
        return pooled

    def _create_intervention_hook(self, steering_vector: torch.Tensor, coeff: float = None):
        """Hook that adds a steering displacement relative to the activation norm.

        displacement = direction * ||activation|| * coeff

        Args:
            steering_vector: (hidden,) for a single shared vector, or
                             (batch, hidden) for per-sample vectors.
            coeff: scaling coefficient. Defaults to self.intervention_coeff.
        """
        if coeff is None:
            coeff = self.intervention_coeff

        if steering_vector.dim() == 1:
            direction = steering_vector.unsqueeze(0).unsqueeze(0)  # (1,1,H)
        else:
            direction = steering_vector.unsqueeze(1)  # (B,1,H)
        # Normalize to unit direction
        direction = direction / (direction.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        def intervention_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Scale steering by per-token activation norm
            scale = h.norm(p=2, dim=-1, keepdim=True) * coeff  # (B,S,1)
            modified = h + direction * scale
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return intervention_hook

    def _forward_with_intervention(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        steering_vectors: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass with intervention at layer(s) ℓ."""

        if isinstance(steering_vectors, torch.Tensor):
            steering_vectors = [steering_vectors]

        # Register hooks for all intervention layers
        handles = []
        try:
            for layer_idx, r_ell in zip(self.intervention_layers, steering_vectors):
                intervention_hook = self._create_intervention_hook(r_ell)
                layer_module = self._get_layer_module(model, layer_idx)
                handle = layer_module.register_forward_hook(intervention_hook)
                handles.append(handle)

            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            if hasattr(outputs, "logits"):
                return outputs.logits
            if isinstance(outputs, tuple):
                return outputs[0]
            return outputs
        finally:
            for handle in handles:
                handle.remove()

    def _forward_with_cache(self, model, inputs, layer_idx, no_grad=True):
        """Forward pass caching raw activations at the given layer (RMU-style)."""
        cache = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)

        layer_module = self._get_layer_module(model, layer_idx)
        hook_handle = layer_module.register_forward_hook(hook)
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
        hook_handle.remove()
        return cache[0], outputs

    @staticmethod
    def _compute_activation_loss(activation1, activation2, mask):
        """MSE loss between activations, masked by valid tokens (RMU-style)."""
        squared_diff = F.mse_loss(activation1, activation2, reduction="none")  # (b, s, d)
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)  # (b, s, d)
        squared_diff_sum = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)  # (b,)
        num_tokens = mask.sum(dim=-1, keepdim=True)  # (b, 1)
        return (squared_diff_sum / num_tokens.squeeze(-1)).mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        """Dispatch to phase-specific loss computation."""
        if self._phase == 1:
            return self._compute_loss_phase1(model, inputs, return_outputs)
        else:
            return self._compute_loss_phase2(model, inputs, return_outputs)

    def _compute_loss_phase1(self, model, inputs, return_outputs=False):
        """
        Phase 1: Train encoder to produce steering vectors.

        Total loss:
          L = L_forget + λ L_util + μ L_orth + ρ L_norm

        Forget objective is *suppression* of the provided forget labels,
        i.e., minimize log p(y_f | x_f, +r_ℓ). Implemented as NEGATIVE CE:
          L_forget = - CE(logits, y_f)
        so minimizing L_forget decreases p(y_f).
        """
        # Ensure mapping networks are in training mode
        self.encoder.train()
        self.mapping_network.train()
        if self.attention_pooling is not None:
            self.attention_pooling.train()

        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        forget_batch = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs.get("attention_mask", None),
        }
        retain_batch = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs.get("attention_mask", None),
        }

        # ---- Step 1: Summarize forget set (frozen activations) ----
        # Use the first intervention layer for activation extraction
        with torch.no_grad():
            h_forget = self._extract_layer_activations(
                self.model, forget_batch, self.intervention_layers[0], requires_grad=False
            )

        # Ensure mapping networks are on the same device as activations
        device = h_forget.device
        if next(self.encoder.parameters()).device != device:
            self.encoder.to(device)
            self.mapping_network.to(device)
            if self.attention_pooling is not None:
                self.attention_pooling.to(device)

        z_f = self.encoder(h_forget)           # (batch, latent_dim)
        r_ells = self.mapping_network(z_f)     # (batch, hidden) or list of (batch, hidden)

        # Normalize to list for consistent handling
        if isinstance(r_ells, torch.Tensor):
            r_ells_list = [r_ells]
        else:
            r_ells_list = r_ells

        # ---- Step 2: Update online PCA with retain activations ----
        if self.online_pca is not None:
            with torch.no_grad():
                h_retain = self._extract_layer_activations(
                    self.model, retain_batch, self.intervention_layers[0], requires_grad=False
                )
                self.online_pca.update(h_retain)

        # ---- Step 3: Forget loss — MSE between frozen activations and r_ells target ----
        # Train encoder so r_ells becomes the misdirection target for forget activations.
        # Model is frozen, so gradients flow only through r_ells -> encoder.
        layer_idx = self.intervention_layers[0]
        with torch.no_grad():
            model_forget_activations, _ = self._forward_with_cache(
                self.model, forget_batch, layer_idx, no_grad=True
            )
            model_forget_activations = model_forget_activations.detach().float()

        # Expand per-sample r_ells to match activation shape (batch, seq, hidden)
        control_vec = r_ells_list[0].float().unsqueeze(1)  # (batch, 1, hidden)
        control_vec = control_vec.expand_as(model_forget_activations)

        forget_mask = forget_inputs.get("labels", forget_inputs["input_ids"]) != -100
        loss_forget = self._compute_activation_loss(
            model_forget_activations, control_vec, forget_mask
        )

        # ---- Step 4: Utility preservation on retain set: KL(p_base || p_steer) ----
        with torch.no_grad():
            outputs_base = self.model(**{k: v for k, v in retain_batch.items() if k != "labels"})
            logits_base = outputs_base.logits if hasattr(outputs_base, "logits") else (outputs_base[0] if isinstance(outputs_base, tuple) else outputs_base)

        # Use mean steering vector across forget batch for retain utility check
        if isinstance(r_ells, torch.Tensor):
            r_ells_mean = r_ells.mean(dim=0)   # (hidden,)
        else:
            r_ells_mean = [r.mean(dim=0) for r in r_ells]  # list of (hidden,)
        logits_steer = self._forward_with_intervention(self.model, retain_batch, r_ells_mean)

        if self.kl_last_token_only:
            # Memory-friendly: KL only on last token distribution
            logits_base_ = logits_base[:, -1, :].detach()
            logits_steer_ = logits_steer[:, -1, :]
            p_base = F.softmax(logits_base_, dim=-1)
            loss_util = F.kl_div(
                F.log_softmax(logits_steer_, dim=-1),
                p_base,
                reduction="batchmean",
                log_target=False,
            )
        else:
            # Full KL over all tokens (can be expensive)
            p_base = F.softmax(logits_base.detach(), dim=-1)
            loss_util = F.kl_div(
                F.log_softmax(logits_steer, dim=-1),
                p_base,
                reduction="batchmean",
                log_target=False,
            )

        # ---- Step 5: Localization (orthogonality) ----
        # Use pre-computed basis or online PCA basis
        pca_basis = self.retain_pca_basis
        if pca_basis is None and self.online_pca is not None:
            pca_basis = self.online_pca.get_basis()

        if pca_basis is not None:
            # Mean orthogonality loss across batch and layers
            loss_orth = r_ells_list[0].new_zeros(1).squeeze()
            for r_ell in r_ells_list:
                # r_ell: (batch, hidden), pca_basis: (hidden, n_components)
                projected = torch.matmul(r_ell, pca_basis)  # (batch, n_components)
                loss_orth = loss_orth + torch.mean(torch.sum(projected ** 2, dim=-1))
        else:
            loss_orth = r_ells_list[0].new_zeros(1).squeeze() * 0.0

        # ---- Step 6: Norm regularization (mean per-sample norm) ----
        loss_norm = r_ells_list[0].new_zeros(1).squeeze()
        for r_ell in r_ells_list:
            # r_ell: (batch, hidden) — mean of per-sample squared norms
            loss_norm = loss_norm + torch.mean(torch.sum(r_ell ** 2, dim=-1))

        # ---- Step 7: Total loss ----
        loss = loss_forget + self.lambda_util * loss_util + self.mu_orth * loss_orth + self.rho_norm * loss_norm

        # Debug logging (every 100 steps)
        if self.state.global_step % 100 == 0:
            r_norm_avg = sum(r.detach().norm(p=2, dim=-1).mean().item() for r in r_ells_list) / len(r_ells_list)
            logger.info(
                f"[LatentUnlearning Phase 1] step={self.state.global_step}, "
                f"layers={self.intervention_layers}, "
                f"r_norm_avg={r_norm_avg:.4f}, "
                f"loss_forget={loss_forget.detach().item():.4f}, "
                f"loss_util={loss_util.detach().item():.4f}, "
                f"loss_orth={loss_orth.detach().item():.4f}, "
                f"loss_norm={loss_norm.detach().item():.4f}"
            )

        if return_outputs:
            outputs = {
                "loss_forget": loss_forget.detach(),
                "loss_util": loss_util.detach(),
                "loss_orth": loss_orth.detach(),
                "loss_norm": loss_norm.detach(),
                "r_norm": sum(r.detach().norm(p=2) for r in r_ells_list) / len(r_ells_list),
            }
            return loss, outputs
        return loss

    def _compute_forget_loss_ga_entropy(self, model, forget_batch, forget_mask):
        """Gradient ascent + entropy regularization on forget samples.

        Returns (forget_loss, ga_loss, entropy_loss, outputs) for logging.
        """
        outputs = model(**forget_batch)
        logits = outputs.logits  # (batch, seq, vocab)
        labels = forget_batch["labels"]  # (batch, seq)

        # 1. Gradient ascent: reduce P(correct answer)
        nll_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        ga_loss = -nll_loss

        # 2. Entropy maximization: keep outputs diverse, prevent degeneration
        probs = F.softmax(logits, dim=-1)
        token_entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)  # (batch, seq)
        entropy_loss = -(token_entropy * forget_mask.float()).sum() / forget_mask.sum().clamp(min=1)

        forget_loss = ga_loss + self.entropy_weight * entropy_loss
        return forget_loss, ga_loss, entropy_loss, outputs

    def _compute_loss_phase2(self, model, inputs, return_outputs=False):
        """
        Phase 2: Unlearning with configurable forget loss.

        forget_loss_type="mse": RMU-style activation misdirection toward encoder target
        forget_loss_type="ga_entropy": Gradient ascent + entropy regularization on outputs
        forget_loss_type="retain_swap": Push forget activations toward shuffled retain activations
        forget_loss_type="centroid_steer": Bounded steering toward retain activation centroid

        Both modes use the same retain loss: MSE(model_activations, ref_activations).

        Inputs are flat forget tensors (Phase 2 uses forget-only dataloader).
        Retain data is randomly sampled from the precomputed cache.
        """
        # Phase 2 dataloader yields flat forget batches (not nested)
        forget_batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        layer_idx = self.intervention_layers[0]
        forget_mask = inputs["labels"] != -100
        device = inputs["input_ids"].device

        # ---- Sample cached retain batch ----
        cache_idx = torch.randint(0, len(self._cached_retain_batches), (1,)).item()
        cached = self._cached_retain_batches[cache_idx]
        retain_batch = {
            "input_ids": cached["input_ids"].to(device),
            "attention_mask": cached["attention_mask"].to(device),
            "labels": cached["labels"].to(device),
        }
        ref_retain_activations = cached["ref_activations"].to(device)

        # ---- Retain activations from training model ----
        model_retain_activations, _ = self._forward_with_cache(
            model, retain_batch, layer_idx, no_grad=False
        )

        # ---- Forget loss (mode-dependent) ----
        if self.forget_loss_type == "ga_entropy":
            forget_loss, ga_loss, entropy_loss, forget_outputs = (
                self._compute_forget_loss_ga_entropy(model, forget_batch, forget_mask)
            )
        elif self.forget_loss_type == "retain_swap":
            # Push forget activations toward shuffled retain activations
            model_forget_activations, forget_outputs = self._forward_with_cache(
                model, forget_batch, layer_idx, no_grad=False
            )
            # Use ref model's retain activations as target, shuffled per sample
            swap_target = ref_retain_activations.detach().clone()
            batch_size = swap_target.size(0)
            if batch_size > 1:
                perm = torch.randperm(batch_size, device=swap_target.device)
                # Ensure no sample maps to itself
                for i in range(batch_size):
                    if perm[i] == i:
                        swap_idx = (i + 1) % batch_size
                        perm[i], perm[swap_idx] = perm[swap_idx].clone(), perm[i].clone()
                swap_target = swap_target[perm]
            swap_target = swap_target.to(
                dtype=model_forget_activations.dtype,
                device=model_forget_activations.device,
            )
            # Align sequence lengths: pad shorter tensor with zeros (masked out anyway)
            f_seq = model_forget_activations.size(1)
            s_seq = swap_target.size(1)
            if f_seq != s_seq:
                target_seq = f_seq  # match forget activations length
                if s_seq < target_seq:
                    pad = torch.zeros(
                        batch_size, target_seq - s_seq, swap_target.size(2),
                        dtype=swap_target.dtype, device=swap_target.device,
                    )
                    swap_target = torch.cat([swap_target, pad], dim=1)
                else:
                    swap_target = swap_target[:, :target_seq]
            # Compute loss in float32 to avoid bf16 precision issues
            forget_loss = self._compute_activation_loss(
                model_forget_activations.float(),
                swap_target.float(),
                forget_mask,
            )
        elif self.forget_loss_type == "norm_capped":
            # Norm-capped steering: push forget activations a bounded distance
            # toward shuffled retain activations. Each token's displacement is
            # capped at max_steering_norm, giving a consistent perturbation
            # signal without wild variance across samples.
            model_forget_activations, forget_outputs = self._forward_with_cache(
                model, forget_batch, layer_idx, no_grad=False
            )
            # Get ref model's forget activations as the anchor point
            ref_forget_activations, _ = self._forward_with_cache(
                self.ref_model, forget_batch, layer_idx, no_grad=True
            )
            # Get shuffled retain activations as the direction target
            swap_target = ref_retain_activations.detach().clone()
            batch_size = swap_target.size(0)
            if batch_size > 1:
                perm = torch.randperm(batch_size, device=swap_target.device)
                for i in range(batch_size):
                    if perm[i] == i:
                        swap_idx = (i + 1) % batch_size
                        perm[i], perm[swap_idx] = perm[swap_idx].clone(), perm[i].clone()
                swap_target = swap_target[perm]

            # Work in float32 for precision
            ref_forget_f = ref_forget_activations.detach().float()
            swap_target_f = swap_target.float()

            # Align sequence lengths to match forget activations
            f_seq = ref_forget_f.size(1)
            s_seq = swap_target_f.size(1)
            if f_seq != s_seq:
                if s_seq < f_seq:
                    pad = torch.zeros(
                        batch_size, f_seq - s_seq, swap_target_f.size(2),
                        dtype=swap_target_f.dtype, device=swap_target_f.device,
                    )
                    swap_target_f = torch.cat([swap_target_f, pad], dim=1)
                else:
                    swap_target_f = swap_target_f[:, :f_seq]

            # Compute direction from forget anchor toward retain target
            direction = swap_target_f - ref_forget_f  # (B, S, H)
            # Per-token norm capping
            delta_norm = direction.norm(dim=-1, keepdim=True)  # (B, S, 1)
            scale = torch.clamp(
                self.max_steering_norm / (delta_norm + 1e-8), max=1.0
            )
            capped_direction = direction * scale  # (B, S, H)
            # Target = original forget activations + bounded step toward retain
            norm_capped_target = ref_forget_f + capped_direction

            forget_loss = self._compute_activation_loss(
                model_forget_activations.float(),
                norm_capped_target.to(device=model_forget_activations.device),
                forget_mask,
            )
        elif self.forget_loss_type == "centroid_steer":
            # Steer forget activations toward the retain centroid with optional
            # repulsion from the ref model's original forget activations.
            model_forget_activations, forget_outputs = self._forward_with_cache(
                model, forget_batch, layer_idx, no_grad=False
            )
            centroid = self._retain_centroid.to(
                device=model_forget_activations.device,
                dtype=model_forget_activations.dtype,
            )  # (H,)
            # Optional per-sample noise for diversity within the retain distribution
            if self.centroid_noise > 0:
                noise_std = self._retain_std.to(
                    device=centroid.device, dtype=centroid.dtype
                )
                batch_size = model_forget_activations.size(0)
                noise = torch.randn(batch_size, 1, centroid.size(0),
                                    device=centroid.device, dtype=centroid.dtype)
                target = centroid.unsqueeze(0).unsqueeze(0) + self.centroid_noise * noise_std * noise
                target = target.expand_as(model_forget_activations)
            else:
                target = centroid.unsqueeze(0).unsqueeze(0).expand_as(model_forget_activations)

            # Norm capping: bound per-token displacement toward centroid
            with torch.no_grad():
                direction = target - model_forget_activations.detach()
                delta_norm = direction.norm(dim=-1, keepdim=True)
                scale = torch.clamp(
                    self.max_steering_norm / (delta_norm + 1e-8), max=1.0
                )
                capped_target = model_forget_activations.detach() + direction * scale

            attract_loss = self._compute_activation_loss(
                model_forget_activations.float(),
                capped_target.float(),
                forget_mask,
            )

            # Repulsion: push current forget activations away from ref model's
            # original forget activations. Uses hinge loss so repulsion stops
            # once per-token distance exceeds the margin.
            if self.repulsion_weight > 0:
                with torch.no_grad():
                    ref_forget_act, _ = self._forward_with_cache(
                        self.ref_model, forget_batch, layer_idx, no_grad=True
                    )
                ref_forget_f = ref_forget_act.to(
                    device=model_forget_activations.device
                ).float()
                model_forget_f = model_forget_activations.float()
                # Per-token MSE from ref (mean over hidden dim)
                per_token_mse = ((model_forget_f - ref_forget_f) ** 2).mean(dim=-1)  # (B, S)
                # Margin in MSE scale: max_steering_norm^2 / H
                margin = self.max_steering_norm ** 2 / model_forget_f.size(-1)
                hinge = torch.relu(margin - per_token_mse)  # (B, S)
                mask_f = forget_mask.float()
                repulsion_loss = (hinge * mask_f).sum() / (mask_f.sum() + 1e-8)
                forget_loss = attract_loss + self.repulsion_weight * repulsion_loss
            else:
                repulsion_loss = torch.tensor(0.0)
                forget_loss = attract_loss
        else:
            # MSE mode: per-sample encoder control vectors + activation misdirection
            with torch.no_grad():
                h_forget = self._extract_layer_activations(
                    model, forget_batch, layer_idx, requires_grad=False
                )
                z_f = self.encoder(h_forget)           # (batch, latent_dim)
                r_ells = self.mapping_network(z_f)     # (batch, hidden) or list

            if isinstance(r_ells, torch.Tensor):
                control_vec = r_ells
            else:
                control_vec = r_ells[0]

            # Per-sample normalization and scaling
            control_vec = control_vec.detach()  # (batch, hidden)
            control_vec = control_vec / (control_vec.norm(dim=-1, keepdim=True) + 1e-8) * self.steering_coeff
            control_vec = control_vec.unsqueeze(1)  # (batch, 1, hidden)

            model_forget_activations, forget_outputs = self._forward_with_cache(
                model, forget_batch, layer_idx, no_grad=False
            )
            control_vec = control_vec.to(
                dtype=model_forget_activations.dtype,
                device=model_forget_activations.device,
            )
            control_vec = control_vec.expand_as(model_forget_activations)
            attract_loss = self._compute_activation_loss(
                model_forget_activations, control_vec, forget_mask
            )

            # Optional repulsion from ref model's original forget activations
            if self.repulsion_weight > 0:
                with torch.no_grad():
                    ref_forget_act, _ = self._forward_with_cache(
                        self.ref_model, forget_batch, layer_idx, no_grad=True
                    )
                ref_forget_f = ref_forget_act.to(
                    device=model_forget_activations.device
                ).float()
                model_forget_f = model_forget_activations.float()
                per_token_mse = ((model_forget_f - ref_forget_f) ** 2).mean(dim=-1)
                margin = self.max_steering_norm ** 2 / model_forget_f.size(-1)
                hinge = torch.relu(margin - per_token_mse)
                mask_f = forget_mask.float()
                repulsion_loss = (hinge * mask_f).sum() / (mask_f.sum() + 1e-8)
                forget_loss = attract_loss + self.repulsion_weight * repulsion_loss
            else:
                repulsion_loss = torch.tensor(0.0)
                forget_loss = attract_loss

        # ---- Retain loss — anchor activations to reference model ----
        retain_mask = retain_batch["labels"] != -100
        retain_loss = self._compute_activation_loss(
            model_retain_activations,
            ref_retain_activations.to(model_retain_activations.device),
            retain_mask,
        )

        # ---- Total loss (with optional forget warmup) ----
        if self.forget_warmup_steps > 0 and self.state.global_step < self.forget_warmup_steps:
            forget_weight = self.state.global_step / self.forget_warmup_steps
        else:
            forget_weight = 1.0
        loss = forget_weight * forget_loss + self.lambda_util * retain_loss

        # Debug logging (every 100 steps)
        if self.state.global_step % 100 == 0:
            if self.forget_loss_type == "ga_entropy":
                logger.info(
                    f"[LatentUnlearning Phase 2 / GA+Entropy] step={self.state.global_step}, "
                    f"ga_loss={ga_loss.detach().item():.4f}, "
                    f"entropy_loss={entropy_loss.detach().item():.4f}, "
                    f"forget_loss={forget_loss.detach().item():.4f}, "
                    f"retain_loss={retain_loss.detach().item():.4f}, "
                    f"forget_weight={forget_weight:.4f}"
                )
            elif self.repulsion_weight > 0 and self.forget_loss_type in ("centroid_steer", "mse"):
                tag = "CentroidSteer" if self.forget_loss_type == "centroid_steer" else "MSE+Repulsion"
                logger.info(
                    f"[LatentUnlearning Phase 2 / {tag}] step={self.state.global_step}, "
                    f"attract_loss={attract_loss.detach().item():.4f}, "
                    f"repulsion_loss={repulsion_loss.detach().item():.4f}, "
                    f"forget_loss={forget_loss.detach().item():.4f}, "
                    f"retain_loss={retain_loss.detach().item():.4f}, "
                    f"forget_weight={forget_weight:.4f}"
                )
            else:
                tag_map = {"retain_swap": "RetainSwap", "norm_capped": "NormCapped", "centroid_steer": "CentroidSteer"}
                log_tag = tag_map.get(self.forget_loss_type, "RMU")
                logger.info(
                    f"[LatentUnlearning Phase 2 / {log_tag}] step={self.state.global_step}, "
                    f"forget_loss={forget_loss.detach().item():.4f}, "
                    f"retain_loss={retain_loss.detach().item():.4f}, "
                    f"forget_weight={forget_weight:.4f}"
                )

        if return_outputs:
            return loss, forget_outputs
        return loss

    @torch.no_grad()
    def plot_latent_vectors(
        self,
        forget_dataloader,
        retain_dataloader,
        output_path: str = "latent_vectors.png",
        max_batches: int = 20,
        method: str = "pca",
        layer_idx: Optional[int] = None,
    ):
        """
        Plot forget activations, steering vectors, and retain activations in 2D.

        Collects pooled hidden-state vectors from the intervention layer for
        both forget and retain samples, computes the learned steering vector(s),
        and projects everything to 2D for visualization.

        Args:
            forget_dataloader: DataLoader yielding forget-set batches.
            retain_dataloader: DataLoader yielding retain-set batches.
            output_path: Where to save the figure.
            max_batches: Max batches to collect per set (caps sample count).
            method: Dimensionality reduction method ("pca" or "tsne").
            layer_idx: Intervention layer to extract from. Defaults to
                       self.intervention_layers[0].
        """
        import numpy as np
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        if layer_idx is None:
            layer_idx = self.intervention_layers[0]

        device = next(self.model.parameters()).device
        self.model.eval()
        self.encoder.eval()
        self.mapping_network.eval()

        # ---- Collect forget activations ----
        forget_vecs = []
        for i, batch in enumerate(forget_dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"input_ids": batch["input_ids"],
                      "attention_mask": batch.get("attention_mask")}
            h = self._extract_layer_activations(
                self.model, inputs, layer_idx, requires_grad=False
            )
            forget_vecs.append(h.cpu().float())
        forget_vecs = torch.cat(forget_vecs, dim=0).numpy()  # (N_f, hidden)

        # ---- Compute per-sample steering vectors ----
        all_forget_pooled = torch.from_numpy(forget_vecs).to(device)
        z_f = self.encoder(all_forget_pooled)      # (N_f, latent_dim)
        r_ells = self.mapping_network(z_f)         # (N_f, hidden) or list of (N_f, hidden)
        if isinstance(r_ells, torch.Tensor):
            steering_vecs = r_ells.cpu().float().numpy()  # (N_f, hidden)
        else:
            steering_vecs = r_ells[0].cpu().float().numpy()  # (N_f, hidden) — first layer

        # ---- Collect retain activations ----
        retain_vecs = []
        for i, batch in enumerate(retain_dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"input_ids": batch["input_ids"],
                      "attention_mask": batch.get("attention_mask")}
            h = self._extract_layer_activations(
                self.model, inputs, layer_idx, requires_grad=False
            )
            retain_vecs.append(h.cpu().float())
        retain_vecs = torch.cat(retain_vecs, dim=0).numpy()  # (N_r, hidden)

        # ---- Compute forget + steering (steered forget activations) ----
        # Each forget activation displaced by its own per-sample steering vector
        steered_forget_vecs = forget_vecs + steering_vecs  # (N_f, hidden) element-wise

        # ---- Dimensionality reduction ----
        all_vecs = np.concatenate(
            [forget_vecs, steered_forget_vecs, steering_vecs, retain_vecs], axis=0
        )
        n_forget = len(forget_vecs)
        n_steered = len(steered_forget_vecs)
        n_steer = len(steering_vecs)

        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=min(30, len(all_vecs) - 1),
                           random_state=42)
            coords = reducer.fit_transform(all_vecs)
        else:
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(all_vecs)

        idx = 0
        forget_coords = coords[idx:idx + n_forget]; idx += n_forget
        steered_coords = coords[idx:idx + n_steered]; idx += n_steered
        steer_coords = coords[idx:idx + n_steer]; idx += n_steer
        retain_coords = coords[idx:]

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(forget_coords[:, 0], forget_coords[:, 1],
                   c="tab:red", marker="x", s=40, alpha=0.6, label="Forget activations")
        ax.scatter(steered_coords[:, 0], steered_coords[:, 1],
                   c="tab:orange", marker="d", s=40, alpha=0.6, label="Forget + steering")
        ax.scatter(retain_coords[:, 0], retain_coords[:, 1],
                   c="tab:blue", marker="o", s=40, alpha=0.6, label="Retain activations")
        ax.scatter(steer_coords[:, 0], steer_coords[:, 1],
                   c="tab:green", marker="*", s=60, alpha=0.6,
                   zorder=5, label="Steering vectors (per-sample)")

        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(f"Latent Vectors — Layer {layer_idx}")
        ax.legend()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved latent vector plot to %s", output_path)
