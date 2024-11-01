from pytorch_lightning import LightningModule
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import torch
import wandb

class LightningClassifier(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=3  # For sentiment analysis
        )
        self.save_hyperparameters(config.__dict__)
        
    def forward(self, **inputs):
        return self.model(**inputs)
        
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=-1)
        
        return {'val_loss': loss, 'preds': preds, 'labels': batch['labels']}
        
    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu()
        all_labels = torch.cat([x['labels'] for x in outputs]).cpu()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        
        # Log confusion matrix to wandb
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=['Negative', 'Neutral', 'Positive']
            )
        })
        
    def configure_optimizers(self):
        # Separate parameters for weight decay
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() 
                       if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() 
                       if any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(param_groups, lr=self.config.learning_rate)
        
        # Calculate total steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }