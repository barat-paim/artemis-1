# model_utils.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch
from typing import Tuple, Any
from config import TrainingConfig

def setup_model_and_tokenizer(config: TrainingConfig) -> Tuple[Any, Any]:
    """Initialize the model and tokenizer with proper configurations"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        torch_dtype=config.dtype,
        device_map="auto"
    )

    if config.use_lora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
    return model, tokenizer