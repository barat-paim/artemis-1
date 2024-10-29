# monitor.py

import logging
import time
from pathlib import Path
import torch
import json
from typing import Dict, Any
from config import TrainingConfig
import curses
import sys

class TrainingMonitor:
    def __init__(self, config: TrainingConfig, dashboard=None):
        self.config = config
        self.start_time = time.time()
        self.metrics_history = []
        self.setup_logging()
        self.dashboard = dashboard
        
        # Initialize dashboard if in interactive mode
        if sys.stdout.isatty():
            try:
                self.dashboard = curses.wrapper(lambda stdscr: TrainingDashboard(stdscr, config))
            except:
                curses.endwin()  # Ensure terminal is restored

    def setup_logging(self):
        # Create output directory if it doesn't exist
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(Path(self.config.output_dir) / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        gpu_memory = (
            torch.cuda.memory_allocated() / 1024 ** 2
            if torch.cuda.is_available() else 0
        )
       
        metrics_with_meta = {
            "step": step,
            "time_elapsed": time.time() - self.start_time,
            "gpu_memory": gpu_memory,
            **metrics
        }
        self.metrics_history.append(metrics_with_meta)
        self.logger.info(f"Step {step}: {metrics}")
        
        # Update dashboard if available
        if self.dashboard:
            self.dashboard.update_metrics(metrics_with_meta)
    
    # Save metrics to a file
    def save_metrics(self):
        metrics_path = Path(self.config.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f)
    
    def cleanup(self):
        """Cleanup resources and restore terminal"""
        if self.dashboard:
            self.dashboard.cleanup()
            self.dashboard = None