from pathlib import Path
import torch
from typing import List, Dict
import wandb
from config import TrainingConfig

class LightningSentimentPredictor:
    def __init__(self, lightning_model, tokenizer, config: TrainingConfig):
        self.model = lightning_model.model  # Access the underlying HF model
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
            
        result = {
            'text': text,
            'sentiment': self.sentiment_labels[prediction.item()],
            'confidence': probs[0][prediction.item()].item(),
            'probabilities': {
                label: prob.item()
                for label, prob in zip(self.sentiment_labels, probs[0])
            }
        }
        
        # Log prediction to W&B
        wandb.log({
            'inference_examples': wandb.Table(
                columns=['text', 'prediction', 'confidence'],
                data=[[text, result['sentiment'], result['confidence']]]
            )
        })
        
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Predict sentiments for a batch of texts"""
        results = [self.predict(text) for text in texts]
        
        # Create confusion matrix data
        predictions = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Log batch metrics to W&B
        wandb.log({
            'inference_batch': {
                'avg_confidence': sum(confidences) / len(confidences),
                'prediction_distribution': wandb.Histogram([
                    self.sentiment_labels.index(p) for p in predictions
                ])
            }
        })
        
        return results

def test_model(lightning_model, tokenizer, config: TrainingConfig) -> List[Dict]:
    """Test the model with sample texts and log results to W&B"""
    predictor = LightningSentimentPredictor(lightning_model, tokenizer, config)
    
    test_texts = [
        "This movie was amazing!",
        "I didn't like it at all.",
        "It was okay, nothing special.",
        "Best experience ever, highly recommended!",
        "Terrible service and poor quality."
    ]
    
    results = predictor.predict_batch(test_texts)
    
    # Create and log test results table to W&B
    test_table = wandb.Table(
        columns=['text', 'prediction', 'confidence', 'probabilities'],
        data=[
            [r['text'], r['sentiment'], r['confidence'], str(r['probabilities'])]
            for r in results
        ]
    )
    wandb.log({'test_results': test_table})
    
    return results