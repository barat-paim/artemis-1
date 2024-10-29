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
import time

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
        save_steps=10,
        early_stopping_patience=50,
        num_train_samples=1000,
        num_eval_samples=100
    )

    try:
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
            
            # Run final evaluation
            dashboard.set_status("Running final evaluation...")
            final_metrics = trainer.evaluate()
            
            # Update dashboard with final metrics
            final_metrics['is_eval_step'] = True
            monitor.log_metrics(final_metrics, trainer.global_step)
            
            # Save results
            dashboard.set_status("Saving final results...")
            monitor.save_metrics()
            
            # Show completion message
            dashboard.set_status("Training completed successfully!")
            time.sleep(2)  # Give time to see final status
            
            # Save final results before cleanup
            dashboard._save_final_results()
            
            # Run inference tests after dashboard cleanup
            dashboard.set_status("Running inference tests...")
            inference_results = test_model(model, tokenizer, config)
            
            # Update dashboard with inference results
            current_y = len(dashboard.eval_history) + 20  # Approximate position after other sections
            dashboard._draw_inference_results(current_y, 2, inference_results)
            dashboard.stdscr.refresh()
            
            # Show completion message
            dashboard.set_status("Training and inference completed successfully!")
            time.sleep(5)  # Give more time to see inference results
            
            # Save final results before cleanup
            dashboard._save_final_results()
            
            # Setup keyboard handling
            stdscr.nodelay(1)  # Non-blocking input
            while True:
                try:
                    key = stdscr.getch()
                    if key == ord('q'):
                        dashboard.set_status("Stopping training...")
                        break
                    elif key == ord('s'):
                        dashboard.set_status("Saving checkpoint...")
                        trainer.save_checkpoint()
                        dashboard.set_status("Checkpoint saved")
                except curses.error:
                    pass
            
            # Clean up curses exactly once
            curses.endwin()
            
        except KeyboardInterrupt:
            dashboard.set_status("PROCESS INTERRUPTED BY USER")
        except Exception as e:
            dashboard.set_status(f"Error occurred: {str(e)}")
            time.sleep(2)  # Give time to see error message
            curses.endwin()

    except Exception as e:
        if 'dashboard' in locals():
            dashboard.set_status(f"Error occurred: {str(e)}")
            time.sleep(2)  # Give time to see error message
            curses.endwin()
        raise e

def main():
    """
    Main entry point that wraps the training function with curses
    """
    curses.wrapper(run_training)

if __name__ == "__main__":
    main()