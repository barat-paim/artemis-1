import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import bitsandbytes as bnb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_tensorboard():
    """Setup TensorBoard logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = f"./runs/alpaca_finetune_{timestamp}"
    writer = SummaryWriter(tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    return tensorboard_dir

def load_tokenizer_and_model(model_path):
    """Load and prepare the model and tokenizer"""
    logger.info("Loading model and tokenizer...")
    logger.info(f"Before loading model: {get_gpu_memory()}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.float16
    )
    
    logger.info(f"After loading model: {get_gpu_memory()}")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,                     # Rank
        lora_alpha=32,           # Alpha scaling
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    logger.info(f"After LoRA setup: {get_gpu_memory()}")
    
    return tokenizer, model

def prepare_dataset(tokenizer, max_length=512):
    """Load and prepare the dataset"""
    logger.info("Loading dataset...")
    dataset = load_from_disk("./data/alpaca_prepared")
    
    def tokenize_function(examples):
        # Combine instruction, input, and output
        texts = []
        for i in range(len(examples['instruction'])):
            text = examples['text'][i]
            if examples['input'][i]:
                text += examples['input'][i] + "\n"
            text += examples['output'][i]
            texts.append(text)
        
        # Tokenize with padding and truncation
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Tokenize datasets
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def get_gpu_memory():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    return "GPU not available"

def main():
    # Setup
    model_path = "./model/llama_3_2_1b_model"
    tensorboard_dir = setup_tensorboard()
    
    # Load model and tokenizer
    tokenizer, model = load_tokenizer_and_model(model_path)
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    logger.info(f"Training on {len(dataset['train'])} examples, "
                f"evaluating on {len(dataset['validation'])} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results_alpaca',
        num_train_epochs=3,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        gradient_accumulation_steps=4,
        eval_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        logging_dir=tensorboard_dir,
        report_to=["tensorboard"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Before training: {get_gpu_memory()}")
    trainer.train()
    
    # Save final model
    logger.info("Saving model...")
    trainer.save_model("./fine_tuned_llama_alpaca")
    
    # Close TensorBoard writer
    trainer.state.log_history[-1].pop("train_runtime", None)
    trainer.state.log_history[-1].pop("train_samples_per_second", None)
    trainer.state.log_history[-1].pop("train_steps_per_second", None)
    trainer.state.log_history[-1].pop("train_loss", None)
    
if __name__ == "__main__":
    main()