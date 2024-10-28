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

# Initialize TensorBoard writer
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_dir = os.path.join("./runs", f"squad_finetune_{run_name}")
writer = SummaryWriter(tensorboard_dir)
logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")

# ... (ImprovedSQuADDataset class remains the same) ...

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
