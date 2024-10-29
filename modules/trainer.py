# trainer.py
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    subprocess.check_call(["pip3", "install", "scikit-learn", "--user"])
    from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Optional
from config import TrainingConfig
from monitor import TrainingMonitor
from datasets import Dataset
from tqdm import tqdm

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
        self.best_metric = float('-inf')  # For tracking best F1 score
        self.setup_optimization()
        self.patience = config.early_stopping_patience
        self.best_loss = float('inf')
        self.no_improve_count = 0
        self.gradient_history = []

    def setup_optimization(self):
        """Initialize optimizer and learning rate scheduler with weight decay handling"""
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() 
                       if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() 
                       if any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': 0.0}
        ]
        
        self.optimizer = AdamW(param_groups, lr=self.config.learning_rate)
        
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def evaluate(self):
        """Evaluate the model using sklearn metrics"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics using sklearn
        metrics = {
            'eval_loss': total_loss / len(self.eval_dataloader),
            'eval_accuracy': accuracy_score(all_labels, all_preds),
            'eval_f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        # Save best model if F1 score improved
        if metrics['eval_f1'] > self.best_metric:
            self.best_metric = metrics['eval_f1']
            self.save_checkpoint('best_model')
        
        self.model.train()
        return metrics

    def save_checkpoint(self, name: str):
        """Save a checkpoint of the model with additional training state"""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'step': self.global_step if hasattr(self, 'global_step') else 0,
        }, checkpoint_dir / "training_state.pt")

    def _compute_gradient_norm(self):
        """Compute total gradient norm across all parameters"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def train(self):
        """Train the model with improved logging and monitoring"""
        self.model.train()
        self.global_step = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            
            for batch in self.train_dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                
                # Compute gradient norm before clipping
                gradient_norm = self._compute_gradient_norm()
                self.gradient_history.append(gradient_norm)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                # Enhanced metrics logging
                if self.monitor and self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    metrics = {
                        'loss': loss.item(),
                        'learning_rate': current_lr,
                        'epoch': epoch,
                        'gradient_norm': gradient_norm,
                        'steps_without_improvement': self.no_improve_count
                    }
                    self.monitor.log_metrics(metrics, self.global_step)
                
                # Early stopping check
                if loss.item() < self.best_loss:
                    self.best_loss = loss.item()
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    
                if self.no_improve_count >= self.patience:
                    print("\nEarly stopping triggered!")
                    return
                    
                self.global_step += 1
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            if self.monitor:
                self.monitor.log_metrics({'epoch_loss': avg_epoch_loss}, self.global_step)
