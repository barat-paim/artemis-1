# main.py

from pathlib import Path
import torch
import curses
from config import TrainingConfig
from model_utils import setup_model_and_tokenizer
from monitor import TrainingMonitor
from trainer import Trainer
from dataloader import TextClassificationDataset
from datasets import load_dataset
from inference import test_model
from dashboard import TrainingDashboard

def run_training(stdscr):
    # initialize config
    config = TrainingConfig(
        model_name="facebook/opt-125m",
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
        max_steps=1000,
        save_steps=100,
        early_stopping_patience=50
    )

    # Initialize dashboard first
    dashboard = TrainingDashboard(stdscr, config)
    
    # Initialize monitor with dashboard
    monitor = TrainingMonitor(config, dashboard)

    dashboard.set_status(f"Starting New Experiment with Model: {config.model_name}")

    # Load dataset
    try:
        dashboard.set_status("Loading dataset...")
        dataset = load_dataset("tweet_eval", "sentiment")
        train_data = dataset['train'].select(range(config.train_size))
        eval_data = dataset['test'].select(range(config.eval_size))
        dashboard.set_status(f"Dataset loaded with {len(train_data)} training samples")
    except Exception as e:
        dashboard.set_status(f"Error loading dataset: {str(e)}")
        return

    # setup model and tokenizer
    try:
        dashboard.set_status("Setting up MODEL and TOKENIZER...")
        model, tokenizer = setup_model_and_tokenizer(config)
        dashboard.set_status("SETUP COMPLETED")
    except Exception as e:
        dashboard.set_status(f"Error setting up model and tokenizer: {str(e)}")
        return

    # Prepare Datasets
    dashboard.set_status("Preparing TRAIN and EVAL DATASETS...")
    train_dataset = TextClassificationDataset(train_data, tokenizer, config)
    eval_dataset = TextClassificationDataset(eval_data, tokenizer, config)
    dashboard.set_status("DATASETS PREPARED")

    # Initialize trainer and start training
    dashboard.set_status("Initializing TRAINER...")
    dashboard.set_status("well, here we load the model, config, train and eval datasets, and the monitor")
    trainer = Trainer(model, config, train_dataset, eval_dataset, monitor)

    try:
        dashboard.set_status("Starting TRAINING...")
        trainer.train()
        dashboard.set_status("TRAINING COMPLETED")
    except KeyboardInterrupt:
        dashboard.set_status("PROCESS INTERRUPTED BY USER")
    except Exception as e:
        dashboard.set_status(f"Error occurred: {str(e)}")
    finally:
        if 'monitor' in locals():
            monitor.cleanup()

    # Test model predictions
    dashboard.set_status("Testing model predictions...")
    test_model(model, tokenizer, config)

def main():
    """
    Main entry point that wraps the training function with curses
    """
    curses.wrapper(run_training)

if __name__ == "__main__":
    main()