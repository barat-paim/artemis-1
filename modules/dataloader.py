# dataloader.py
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List
import torch
from config import TrainingConfig

class BaseDataset(Dataset):
    def __init__(self, data, tokenizer, config: TrainingConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Implement this method in child classes")
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Implement this method in child classes") 
    
class TextClassificationDataset(BaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input text
        inputs = self.tokenizer(
            item['text'],
            max_length=self.config.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch])
        }