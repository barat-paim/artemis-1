# main.py

from pathlib import Path
import torch

from config import TrainingConfig
from model_utils import setup_model_and_tokenizer
from monitor import TrainingMonitor
from trainer import Trainer
from dataloader import TextClassificationDataset
from datasets import load_dataset
from inference import test_model

def main():
    # initialize config
    config = TrainingConfig(
        model_name="facebook/opt-125m", # path to the model
        train_size=1000,
        eval_size=100,
        batch_size=16,
        num_epochs=3,
        model_max_length=512,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        output_dir="./test_run",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nStarting experiment with model: {config.model_name}")
    print(f"Training on device: {config.device}\n")

    # Load dataset
    try:
        print("Loading dataset...")
        dataset = load_dataset("tweet_eval", "sentiment")
        train_data = dataset['train'].select(range(config.train_size))
        eval_data = dataset['test'].select(range(config.eval_size))
        print(f"Dataset loaded successfully with {len(train_data)} training samples and {len(eval_data)} evaluation samples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # setup model and tokenizer
    try:
        print("\nSetting up model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(config)
        print("Model and tokenizer setup successfully")
    except Exception as e:
        print(f"Error setting up model and tokenizer: {str(e)}")
        return

    # Initialize monitor
    print("\nInitializing monitor...")
    monitor = TrainingMonitor(config)
    print("Monitor initialized successfully")

    # Prepare Datasets
    print("\nPreparing datasets...")
    train_dataset = TextClassificationDataset(train_data, tokenizer, config)
    eval_dataset = TextClassificationDataset(eval_data, tokenizer, config)
    print("Datasets prepared successfully")

    # Initialize trainer and start training
    print("\nInitializing trainer...")
    trainer = Trainer(model, config, train_dataset, eval_dataset, monitor)

    try:
        trainer.train()
        print("Training completed successfully")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return

    # Test model predictions
    print("*" * 20)
    print("\nTesting model predictions...")
    test_model(model, tokenizer, config)

if __name__ == "__main__":
    main()