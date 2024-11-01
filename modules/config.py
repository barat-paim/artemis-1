# config.py
from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class TrainingConfig:
    # Model Settings
    model_name: str
    model_max_length: int = 512

    # Training Settings
    batch_size: int = 8
    eval_batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    save_steps: int = 500
    max_steps: int = 1000
    num_train_samples: int = 1000
    num_eval_samples: int = 100
    save_top_k: int = 3 # Added Recently

    # Hardware Settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16

    # Dataset Settings
    train_size: int = 1000
    eval_size: int = 100

    # Logging & Checkpointing
    output_dir: str = "./lightning_outputs"
    logging_steps: int = 10
    eval_steps: int = 20

    # LoRA Settings (if using LoRA)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    # Early Stopping
    early_stopping_patience: int = 50  # Number of steps without improvement before stopping

    # Wandb Configuration
    wandb_project: str = "sentiment-analysis"
    wandb_entity: Optional[str] = None
    gradient_clip_val: float = 1.0

