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
    # Initialize config (keep existing config setup)
    config = TrainingConfig(
        model_name="facebook/opt-125m",
        train_size=1000,
        eval_size=100,
        batch_size=12,
        num_epochs=3,
        model_max_length=512,
        learning_rate=5e-6, # consider reducing this 
        weight_decay=0.01,
        warmup_ratio=0.1,
        output_dir="./test_run",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=1000,
        save_steps=100,
        early_stopping_patience=50,
        num_train_samples=1000,  # Added
        num_eval_samples=100     # Added
    )

    try:
        # Initialize dashboard and monitor (keep existing initialization)
        dashboard = TrainingDashboard(stdscr, config)
        monitor = TrainingMonitor(config, dashboard)
        dashboard.set_status(f"Starting New Experiment with Model: {config.model_name}")

        # Setup phase
        try:
            # Load dataset and setup model (keep existing setup code)
            dataset = load_dataset("tweet_eval", "sentiment")
            train_data = dataset['train'].select(range(config.train_size))
            eval_data = dataset['test'].select(range(config.eval_size))
            model, tokenizer = setup_model_and_tokenizer(config)
            train_dataset = TextClassificationDataset(train_data, tokenizer, config)
            eval_dataset = TextClassificationDataset(eval_data, tokenizer, config)
            trainer = Trainer(model, config, train_dataset, eval_dataset, monitor)
        except Exception as e:
            dashboard.set_status(f"Setup error: {str(e)}")
            return handle_exit(dashboard)

        # Training phase
        try:
            dashboard.set_status("Starting TRAINING...")
            trainer.train()
            
            # Evaluation phase
            dashboard.set_status("Running final evaluation...")
            final_metrics = trainer.evaluate()
            final_metrics['is_eval_step'] = True
            monitor.log_metrics(final_metrics, trainer.global_step)
            
            # Results phase
            dashboard.set_status("Saving results...")
            monitor.save_metrics()
            dashboard._save_final_results()
            
            # Inference phase
            dashboard.set_status("Running inference tests...")
            inference_results = test_model(model, tokenizer, config)
            current_y = len(dashboard.eval_history) + 20
            dashboard._draw_inference_results(current_y, 2, inference_results)
            dashboard.stdscr.refresh()
            
            dashboard.set_status("Training completed! Press 'q' to exit, 's' to save checkpoint")
            
        except KeyboardInterrupt:
            dashboard.set_status("Training interrupted! Press 'q' to exit, 's' to save checkpoint")
        except Exception as e:
            dashboard.set_status(f"Error: {str(e)}! Press 'q' to exit")

        # Interactive dashboard phase
        return handle_dashboard_interaction(dashboard, trainer)

    except Exception as e:
        if 'dashboard' in locals():
            dashboard.set_status(f"Critical error: {str(e)}")
            return handle_exit(dashboard)
        raise e

def handle_dashboard_interaction(dashboard, trainer):
    """Handle user interaction with the dashboard"""
    stdscr = dashboard.stdscr
    stdscr.nodelay(0)  # Make getch blocking
    
    while True:
        try:
            key = stdscr.getch()
            if key == ord('q'):
                dashboard.set_status("Exiting...")
                time.sleep(1)
                return handle_exit(dashboard)
            elif key == ord('s'):
                dashboard.set_status("Saving checkpoint...")
                checkpoint_path, latest_path = trainer.save_checkpoint()
                dashboard._show_checkpoint_info(checkpoint_path, latest_path)
                dashboard.set_status("Checkpoint saved. Press 'q' to exit")
            dashboard.stdscr.refresh()
        except curses.error:
            continue

def handle_exit(dashboard):
    """Handle clean exit from the dashboard"""
    try:
        dashboard._save_final_results()
        time.sleep(1)  # Give time to see final message
        curses.endwin()
        return True
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return False

def main():
    """Main entry point"""
    success = curses.wrapper(run_training)
    if not success:
        print("Program terminated with errors. Check logs for details.")

if __name__ == "__main__":
    main()