# monitor.py

import logging
import time
from pathlib import Path
import torch
import json
from typing import Dict, Any

class TrainingMonitor:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.start_time = time.time()
        self.metrics_history = []
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
            handlers=[
                logging.FileHandler(self.config.output_dir / "training.log"),
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
    
    # Save metrics to a file
    def save_metrics(self):
        metrics_path = Path(self.config.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f)