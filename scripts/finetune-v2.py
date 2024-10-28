import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_from_disk
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import time
from torch.utils.data import Dataset
import logging
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add this function near the top of the file, after imports
def monitor_gpu():
    """Monitor GPU memory usage."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved"
    return "GPU not available"
# Initialize TensorBoard writer
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_dir = os.path.join("./runs", f"squad_finetune_{run_name}")
writer = SummaryWriter(tensorboard_dir)
logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")

class ImprovedSQuADDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Improved prompt template with clearer structure
        input_text = (
            "Answer the question based on the given context. "
            "Provide a short, direct answer.\n\n"
            f"Context: {example['context']}\n"
            f"Question: {example['question']}\n"
            "Answer:"
        )
        
        # Add explicit end token to help model learn answer boundaries
        target_text = f" {example['answers']['text'][0]}</s>" if example['answers']['text'] else ""

        # Tokenize with improved handling
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=50,  # Shorter max length for answers
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }


class TensorBoardCallback:
    def __init__(self, writer):
        self.writer = writer
        self.train_step = 0
        self.eval_step = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Log training metrics
        if 'loss' in logs:
            self.writer.add_scalar('train/loss', logs['loss'], self.train_step)
            if 'learning_rate' in logs:
                self.writer.add_scalar('train/learning_rate', logs['learning_rate'], self.train_step)
            if 'grad_norm' in logs:
                self.writer.add_scalar('train/grad_norm', logs['grad_norm'], self.train_step)
            self.train_step += 1
        
        # Log evaluation metrics
        if 'eval_loss' in logs:
            self.writer.add_scalar('eval/loss', logs['eval_loss'], self.eval_step)
            if 'eval_runtime' in logs:
                self.writer.add_scalar('eval/runtime', logs['eval_runtime'], self.eval_step)
            self.eval_step += 1
        
        # Log memory usage
        if 'gpu_memory' in logs:
            memory_str = logs['gpu_memory']
            if isinstance(memory_str, str) and "GB allocated" in memory_str:
                allocated = float(memory_str.split("GB allocated")[0].split(": ")[-1])
                self.writer.add_scalar('system/gpu_memory_allocated', allocated, self.train_step)

class MetricsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
    
    def log(self, logs):
        logs = logs.copy()
        logs["gpu_memory"] = monitor_gpu()
        
        if self.state.global_step > 0:
            elapsed_time = time.time() - self.start_time
            samples_per_second = (
                self.state.global_step * 
                self.args.per_device_train_batch_size / 
                elapsed_time
            )
            logs["training_samples_per_second"] = samples_per_second
            
            # Log training speed to TensorBoard
            writer.add_scalar(
                'performance/samples_per_second',
                samples_per_second,
                self.state.global_step
            )
        
        super().log(logs)

def main():
    # ... (earlier setup code remains the same) ...
    logger.info("Loading dataset...")
    squad_dataset = load_from_disk("./data/squad")
    train_data = squad_dataset['train'].select(range(100))
    eval_data = squad_dataset['validation'].select(range(20))

    logger.info(f"Training on {len(train_data)} examples, evaluating on {len(eval_data)} examples")
    logger.info(f"Evaluating on {len(eval_data)} examples")

    # Load Model
    logger.info("Loading model...")
    model_path = "./llama_3_2_1b_model"
    # initial monitoring gpu
    logger.info(f"Before loading model: {monitor_gpu()}")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    logger.info(f"After loading model: {monitor_gpu()}")

    # Setup LoRa
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"After LoRA setup: {monitor_gpu()}")

    train_dataset = ImprovedSQuADDataset(train_data, tokenizer)
    eval_dataset = ImprovedSQuADDataset(eval_data, tokenizer)
    
    # Training arguments with TensorBoard logging
    training_args = TrainingArguments(
        output_dir='./results_improved',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_ratio=0.1,
        # TensorBoard specific arguments
        logging_dir=tensorboard_dir,
        report_to=["tensorboard"],
    )
    
    # Initialize trainer with TensorBoard callback
    trainer = MetricsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[TensorBoardCallback(writer)]
    )
    
    try:
        # Train and track metrics
        logger.info("Starting training...")
        logger.info(f"Before training: {monitor_gpu()}")
        trainer.train()
        logger.info(f"After training: {monitor_gpu()}")
        
        # Save the model
        logger.info("Saving model...")
        model.save_pretrained("./fine_tuned_llama_squad_improved")
        tokenizer.save_pretrained("./fine_tuned_llama_squad_improved")
        
    finally:
        # Close TensorBoard writer
        writer.close()
        logger.info("TensorBoard writer closed")

if __name__ == "__main__":
    main()