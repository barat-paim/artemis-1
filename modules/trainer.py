# trainer.py

from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, Trainer, TrainingArguments, TrainerCallback
from typing import Optional
from config import TrainingConfig
from monitor import TrainingMonitor
from datasets import Dataset

# import local modules

# import from Hugging Face

class Trainer:
    def __init__(
            self,
            model,
            config: TrainingConfig,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            monitor: Optional[TrainingMonitor] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn
        ) if eval_dataset else None
        self.monitor = monitor
        self.setup_optimization()
        
    def setup_optimization(self):
        """Initialize optimizer and learning rate scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def save_checkpoint(self, step: int):
        """Save a checkpoint of the model"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)

        # Save optimizer and scheduler states
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step,
        }, checkpoint_dir / "optimizer_scheduler.pth")

    def train(self):
        """Train the model"""
        self.model.train()
        global_step = 0

        # train loop
        for epoch in range(self.config.num_epochs):
            for batch in self.train_dataloader:
                # move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # backward pass
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # log metrics
                if self.monitor and global_step % self.config.logging_steps == 0:
                    self.monitor.log_metrics(
                        {"loss": loss.item()},
                        global_step
                    )
                
                # evaluate model
                if (self.eval_dataloader and
                    global_step % self.config.eval_steps == 0):
                    eval_metrics = self.evaluate()
                    if self.monitor:
                        self.monitor.log_metrics(eval_metrics, global_step)
                
                # save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)
                
                global_step += 1

    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.eval_dataloader)
        self.model.train()
        return {"eval_loss": avg_loss}
