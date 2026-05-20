"""
PURGE baseline (strzar/purge) — thin SteerGRPO alias.

The actual PURGE reward functions live in ``_purge_rewards`` and are
dispatched inside ``SteerGRPO.reward_fn`` when ``reward_type`` is one of
``binary``, ``exponential_decay``, ``pagerank``. This class only forces
the hygiene defaults that turn SteerGRPO into a faithful baseline:
every SteerGRPO-only feature (anti_answer, naturalness, retain reward,
retain NLL, KL, entropy, curriculum, resample_low_var, retain_grpo) is
zeroed out so the only reward signal is the PURGE one.

Reference: https://github.com/strzar/purge
"""

from typing import Optional

from trainer.unlearn.SteerGRPO import SteerGRPO


# SteerGRPO knobs PurgeGRPO refuses to inherit "as configured" — these are
# either our auxiliary signals (zeroed) or our degenerate-group escape
# hatches (turned off). They are silently overridden in __init__.
_FORCED_OFF = {
    "resample_low_var":          False,
    "retain_grpo_weight":        0.0,
    "entropy_beta":              0.0,
    "curriculum":                False,
    "answer_reward_weight":      0.0,
    "naturalness_reward_weight": 0.0,
    "retain_reward_weight":      0.0,
    "retain_loss_weight":        0.0,
}


class PurgeGRPO(SteerGRPO):
    def __init__(
        self,
        *args,
        reward_type: str = "binary",
        kl_beta: float = 0.04,           # TRL GRPO default
        **kwargs,
    ):
        # Strip any caller-supplied values for forced-off knobs so the
        # super().__init__ kwargs cannot accidentally re-enable them.
        for k in _FORCED_OFF:
            kwargs.pop(k, None)
        super().__init__(
            *args,
            reward_type=reward_type,
            kl_beta=kl_beta,
            **_FORCED_OFF,
            **kwargs,
        )
