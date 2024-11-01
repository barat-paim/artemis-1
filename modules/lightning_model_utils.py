from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch
from typing import Tuple
from config import TrainingConfig
from lightning_model import LightningClassifier

def setup_lightning_model(config: TrainingConfig) -> Tuple[LightningClassifier, AutoTokenizer]:
    """Initialize the Lightning model and tokenizer with proper configurations"""
    try:
        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=3,
            torch_dtype=config.dtype,
            device_map="auto"
        )

        # Apply LoRA if configured
        if config.use_lora:
            base_model = prepare_model_for_kbit_training(base_model)
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="SEQ_CLS"
            )
            base_model = get_peft_model(base_model, lora_config)

        # Create Lightning model and assign base model
        lightning_model = LightningClassifier(config)
        lightning_model.model = base_model

        return lightning_model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model and tokenizer: {str(e)}")