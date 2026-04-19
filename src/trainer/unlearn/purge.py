"""
GRPO training script for unlearning using modular reward functions.
Configured with Hydra for flexible hyperparameter management.

Usage:
    # Default run (Confucius with PageRank reward)
    python purge.py

    # Different entity
    python purge.py entity=taylor_swift

    # Different reward function
    python purge.py reward=binary

    # Fast testing config
    python purge.py training=fast

    # Override specific parameters
    python purge.py training.num_epochs=20 training.per_device_train_batch_size=4

    # Multi-run sweep
    python purge.py --multirun entity=confucius,taylor_swift,stephen_king
"""
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import json
import hydra
from omegaconf import DictConfig, OmegaConf

from rewards import RewardFunction, BinaryReward, PageRankWeightedReward, ExponentialDecayReward
from rewards.base import RewardConfig


# Original Implementation: https://github.com/strzar/purge/tree/main

def get_reward_class(reward_type: str) -> type[RewardFunction]:
    """
    Get the reward class based on the reward type string.
    
    Args:
        reward_type: Either "binary" or "pagerank"
        
    Returns:
        The corresponding RewardFunction subclass
    """
    reward_classes = {
        "binary": BinaryReward,
        "pagerank": PageRankWeightedReward,
        "exponential_decay": ExponentialDecayReward
    }
    
    if reward_type not in reward_classes:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available: {list(reward_classes.keys())}"
        )
    
    return reward_classes[reward_type]


def load_model_and_tokenizer(cfg: DictConfig):
    """Load the model and tokenizer from HuggingFace."""
    print(f"Loading model: {cfg.model.hf_model_id}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_model_id)
    return model, tokenizer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
    # Print resolved configuration
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load forget dataset
    print(f"Loading dataset from: {cfg.paths.forget_dataset_file}")
    with open(cfg.paths.forget_dataset_file, "r") as f:
        data = json.load(f)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)
    
    # Get reward class and create config
    reward_class = get_reward_class(cfg.reward.type)
    
    # Collect extra params from reward config (excluding 'type')
    extra_params = {k: v for k, v in cfg.reward.items() if k != "type"}
    
    reward_config = RewardConfig(
        target_entity=cfg.entity.name,
        forget_words_file=cfg.paths.forget_words_file,
        forget_dataset_file=cfg.paths.forget_dataset_file,
        model=model,
        tokenizer=tokenizer,
        extra_params=extra_params if extra_params else None,
    )
    
    # Preprocess reward function (e.g., compute PageRank weights)
    print(f"\n{'=' * 60}")
    print(f"Using reward function: {reward_class.__name__}")
    print(f"{'=' * 60}\n")
    
    reward_class.preprocess(reward_config)
    
    # Prepare dataset
    dataset = Dataset.from_list(data)
    if cfg.training.dataset_size is not None:
        dataset = dataset.select(range(min(cfg.training.dataset_size, len(dataset))))
    print(f"Dataset size: {len(dataset)} samples")

    # Training configuration
    training_args = GRPOConfig(
        output_dir=cfg.paths.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=cfg.training.num_generations,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
    )

    # Create trainer with the modular reward function
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_class.calc_reward,
        args=training_args,
        train_dataset=dataset
    )
    
    print("Started training...")
    trainer.train()
    print("Finished training.")


if __name__ == '__main__':
    main()
    
    
"""
Abstract base class for reward functions used in GRPO unlearning.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Optional, Any, Dict
import json


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    target_entity: str
    forget_words_file: str
    forget_dataset_file: str
    model: Any = None
    tokenizer: Any = None
    # Additional reward-specific parameters
    extra_params: Optional[Dict[str, Any]] = None


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    
    Subclasses must implement:
    - preprocess(): Class method to compute any required preprocessing (e.g., PageRank)
    - calc_reward(): Static method that computes rewards for completions
    
    Usage with GRPOTrainer:
        reward_class = PageRankWeightedReward
        reward_class.preprocess(config)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_class.calc_reward,
            ...
        )
    """
    
    # Class-level state that gets set during preprocess()
    _forget_words: Set[str] = set()
    _forget_set: List[str] = []
    _config: Optional[RewardConfig] = None
    _preprocessed: bool = False
    
    @classmethod
    def load_forget_words(cls, forget_words_file: str) -> Set[str]:
        """Load forget words from JSON file."""
        with open(forget_words_file, 'r') as f:
            return set(json.load(f))
    
    @classmethod
    def load_forget_set(cls, forget_words_file: str, target_entity: str) -> List[str]:
        """Load forget set with target entity as first element."""
        with open(forget_words_file, 'r') as f:
            fts_list = json.load(f)
        # Remove target_entity if exists to avoid duplicates, then prepend it
        fts_list = [term for term in fts_list if term != target_entity]
        return [target_entity] + fts_list
    
    @classmethod
    @abstractmethod
    def preprocess(cls, config: RewardConfig) -> None:
        """
        Preprocess step that runs before training.
        
        This method should compute any required data structures (e.g., PageRank weights)
        and store them as class attributes for use in calc_reward().
        
        Args:
            config: RewardConfig with model, tokenizer, and file paths
        """
        pass
    
    @staticmethod
    @abstractmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        """
        Calculate rewards for a batch of completions.
        
        This method is passed directly to GRPOTrainer as reward_funcs.
        It should be a static method that can access class-level state
        set during preprocess().
        
        Args:
            completions: List of generated text completions
            **kwargs: Additional arguments passed by GRPOTrainer
            
        Returns:
            List of reward scores (typically in [0, 1] range)
        """
        pass
    
    @classmethod
    def get_reward_func(cls):
        """
        Returns the calc_reward method bound to the class state.
        Use this when you need to pass the reward function to GRPOTrainer.
        """
        if not cls._preprocessed:
            raise RuntimeError(
                f"{cls.__name__}.preprocess() must be called before get_reward_func()"
            )
        return cls.calc_reward
    
    @classmethod
    def reset(cls) -> None:
        """Reset class-level state."""
        cls._forget_words = set()
        cls._forget_set = []
        cls._config = None
        cls._preprocessed = False


"""
Binary reward function for GRPO unlearning.
Returns 0.0 if completion contains forbidden words, 1.0 otherwise.
"""
import re
from typing import List

from .base import RewardFunction, RewardConfig


class BinaryReward(RewardFunction):
    """
    Simple binary reward function.
    
    - Returns 0.0 if a completion contains any word from the forget set
    - Returns 1.0 otherwise
    
    No preprocessing required beyond loading the forget words.
    """
    
    _forget_words: set = set()
    _pattern: re.Pattern = None
    _preprocessed: bool = False
    
    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        """
        Load forget words and compile regex pattern.
        
        Args:
            config: RewardConfig with forget_words_file path
        """
        cls._config = config
        cls._forget_words = cls.load_forget_words(config.forget_words_file)
        
        # Pre-compile regex pattern for efficiency
        cls._pattern = re.compile(
            r'\b(?:' + '|'.join(map(re.escape, cls._forget_words)) + r')\b',
            re.IGNORECASE
        )
        cls._preprocessed = True
        
        print(f"[BinaryReward] Loaded {len(cls._forget_words)} forget words")
    
    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        """
        Returns 0.0 if a completion contains any word from the forget dataset,
        otherwise returns 1.0.
        
        Args:
            completions: List of generated completions
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of binary rewards (0.0 or 1.0)
        """
        pattern = BinaryReward._pattern
        return [0.0 if pattern.search(completion) else 1.0 for completion in completions]
