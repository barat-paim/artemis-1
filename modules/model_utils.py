# model_utils.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch
from typing import Tuple, Any
from config import TrainingConfig

def setup_lightning_model(config: TrainingConfig) -> Tuple[Any, Any]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = LightningClassifier(config)
        
        if config.use_lora:
            model.model = prepare_model_for_kbit_training(model.model)
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="SEQ_CLS"
            )
            model.model = get_peft_model(model.model, lora_config)
            
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model and tokenizer: {str(e)}")