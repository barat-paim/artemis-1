from pathlib import Path
import torch
from typing import List, Tuple, Dict
from config import TrainingConfig

class SentimentPredictor:
    def __init__(self, model, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()
        self.sentiment_labels = ['negative', 'neutral', 'positive']

    def predict(self, text: str) -> Dict[str, any]:
        """Predict sentiment for a single text"""
        inputs = self.tokenizer(
            text,
            max_length=self.config.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return {
            'text': text,
            'sentiment': self.sentiment_labels[prediction.item()],
            'confidence': probs[0][prediction.item()].item(),
            'probabilities': {
                label: prob.item()
                for label, prob in zip(self.sentiment_labels, probs[0])
            }
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Predict sentiments for a batch of texts"""
        return [self.predict(text) for text in texts]

def test_model(model, tokenizer, config) -> List[Dict]:
    """Predict sentiments for test texts using the SentimentPredictor class"""
    predictor = SentimentPredictor(model, tokenizer, config)
    test_texts = [
        "This movie was amazing!",
        "I didn't like it at all.",
        "It was okay, nothing special.",
        "Best experience ever, highly recommended!",
        "Terrible service and poor quality."
    ]
    
    return predictor.predict_batch(test_texts)