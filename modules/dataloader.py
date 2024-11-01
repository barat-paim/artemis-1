# dataloader.py
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

class TextClassificationDataModule(LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
    def setup(self, stage=None):
        # Load dataset
        dataset = load_dataset("tweet_eval", "sentiment")
        
        # Select subset of data
        self.train_dataset = dataset['train'].select(range(self.config.train_size))
        self.val_dataset = dataset['test'].select(range(self.config.eval_size))
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
    def collate_fn(self, examples):
        texts = [ex['text'] for ex in examples]
        labels = [ex['label'] for ex in examples]
        
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.model_max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }