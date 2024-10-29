import curses
import time
import numpy as np
from typing import Dict, List, Optional
from config import TrainingConfig
from pathlib import Path

class TrainingDashboard:
    def __init__(self, stdscr, config: TrainingConfig):
        self.stdscr = stdscr
        self.config = config
        self.setup_colors()
        self.metrics_history = []
        self.status_message = ""
        self.max_y, self.max_x = stdscr.getmaxyx()
        self.results_file = Path("./logs/results.md")
        self.results_file.parent.mkdir(exist_ok=True)
        self.training_start_time = time.time()
        self.eval_history = []

    def setup_colors(self):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
    def set_status(self, message: str):
        self.status_message = message
        self._draw_dashboard()
        
    def _draw_section_header(self, y: int, x: int, title: str):
        self.stdscr.addstr(y, x, f"═══ {title} ", curses.color_pair(4) | curses.A_BOLD)
        remaining_width = self.max_x - (x + len(title) + 5)
        self.stdscr.addstr(y, x + len(title) + 5, "═" * remaining_width, curses.color_pair(4))
        
    def cleanup(self):
        curses.endwin()
        self._save_final_results()
        
    def _save_final_results(self):
        if not self.metrics_history:
            return
            
        final_metrics = self.metrics_history[-1]
        with open(self.results_file, "w") as f:
            f.write("# Training Results\n\n")
            f.write("## Final Metrics\n")
            f.write(f"- Loss: {final_metrics.get('loss', 0):.4f}\n")
            f.write(f"- Accuracy: {final_metrics.get('eval_accuracy', 0):.2%}\n")
            f.write(f"- F1 Score: {final_metrics.get('eval_f1', 0):.2%}\n")
            f.write(f"- Training Time: {final_metrics.get('time_elapsed', 0)/60:.1f} minutes\n")
            
    def _draw_dashboard(self):
        self.stdscr.clear()
        
        # Main title
        title = "Model Training Dashboard"
        self.stdscr.addstr(0, (self.max_x - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(4))
        
        # Training Configuration Section
        self._draw_section_header(2, 0, "Training Configuration")
        config_items = [
            f"Model: {self.config.model_name}",
            f"Samples: {self.config.num_train_samples} train, {self.config.num_eval_samples} eval",
            f"Batch Size: {self.config.batch_size}",
            f"Learning Rate: {self.config.learning_rate}",
            f"Max Steps: {self.config.max_steps}",
            f"Save Steps: {self.config.save_steps}"
        ]
        for i, item in enumerate(config_items):
            self.stdscr.addstr(4 + i, 2, item)

        current_y = 4 + len(config_items) + 2  # Start next section after config
        
        if self.metrics_history:
            current_metrics = self.metrics_history[-1]
            
            # Training Progress Section
            self._draw_section_header(current_y, 0, "Training Progress")
            current_y += 2
            
            # Draw loss as speedometer
            self._draw_speedometer(current_y, 2, current_metrics.get('loss', 0), 2.0, "Loss")
            current_y += 1
            
            # Draw LR and gradient norm as simple text
            self.stdscr.addstr(current_y, 2, f"Learning Rate: {current_metrics.get('learning_rate', 0):.6f}")
            current_y += 1
            self._draw_gradient_gauge(current_y, 2, current_metrics.get('gradient_norm', 0))
            current_y += 2
            
            # Evaluation Metrics Section - Show as table
            self._draw_section_header(current_y, 0, "Evaluation History")
            current_y += 1
            headers = ["Step", "Loss", "LR", "Grad Norm"]
            self._draw_eval_table(current_y, headers)
            
            # System Status Section
            status_y = current_y + len(self.eval_history) + 3
            self._draw_section_header(status_y, 0, "System Status")
            status_y += 1
            
            epoch = current_metrics.get('epoch', 0)
            gpu_mem = current_metrics.get('gpu_memory', 0)
            elapsed_time = time.time() - self.training_start_time
            
            self.stdscr.addstr(status_y, 2, f"Epoch: {epoch}")
            self.stdscr.addstr(status_y, 20, f"GPU Memory: {gpu_mem:.0f}MB")
            self.stdscr.addstr(status_y + 1, 2, f"Elapsed Time: {self._format_time(elapsed_time)}")
            self._draw_early_stopping(status_y + 2, 2, 
                current_metrics.get('steps_without_improvement', 0),
                self.config.early_stopping_patience)
            
            # Status Message with Progress Bar
            if self.status_message:
                status_y += 4
                self._draw_section_header(status_y, 0, "Status")
                self._draw_progress_bar(status_y + 1, 2, 
                                      current_metrics.get('current_step', 0),
                                      self.config.max_steps,
                                      self.status_message)
        
        self.stdscr.refresh()

    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics and redraw dashboard"""
        self.metrics_history.append(metrics)
        if metrics.get('is_eval_step', False):
            self.eval_history.append(metrics)
        self._draw_dashboard()

    def _draw_speedometer(self, y: int, x: int, value: float, max_value: float, label: str):
        """Draw a speedometer-style progress bar"""
        width = 30
        filled = int(width * min(value / max_value, 1.0))
        bar = "█" * filled + "░" * (width - filled)
        percentage = value / max_value * 100 if max_value != 0 else 0
        
        self.stdscr.addstr(y, x, f"{label:<10}")
        self.stdscr.addstr(y, x + 10, f"[{bar}]")
        self.stdscr.addstr(y, x + width + 12, f"{value:.4f} ({percentage:.1f}%)")

    def _draw_gradient_gauge(self, y: int, x: int, gradient_norm: float):
        """Draw a gauge showing gradient norm status"""
        status = "GOOD" if gradient_norm < 1.0 else "HIGH" if gradient_norm < 10.0 else "ALERT"
        color = (curses.color_pair(1) if status == "GOOD" 
                else curses.color_pair(3) if status == "HIGH"
                else curses.color_pair(2))
        
        self.stdscr.addstr(y, x, f"Gradient Norm: {gradient_norm:.2f} ")
        self.stdscr.addstr(f"[{status}]", color | curses.A_BOLD)

    def _draw_early_stopping(self, y: int, x: int, steps_without_improvement: int, patience: int):
        """Draw early stopping progress"""
        width = 20
        remaining = patience - steps_without_improvement
        filled = int(width * (steps_without_improvement / patience))
        bar = "█" * filled + "░" * (width - filled)
        
        color = (curses.color_pair(1) if remaining > patience // 2
                else curses.color_pair(3) if remaining > patience // 4
                else curses.color_pair(2))
        
        self.stdscr.addstr(y, x, "Early Stop: ")
        self.stdscr.addstr(f"[{bar}] ")
        self.stdscr.addstr(f"{remaining}/{patience}", color | curses.A_BOLD)

    def _draw_eval_table(self, y: int, headers: List[str]):
        """Draw evaluation metrics history as a table"""
        col_width = 15
        # Draw headers
        for i, header in enumerate(headers):
            self.stdscr.addstr(y, 2 + i * col_width, header, curses.A_BOLD)
        
        # Draw rows
        for row_idx, metrics in enumerate(self.eval_history[-10:]):  # Show last 10 entries
            row_y = y + row_idx + 1
            self.stdscr.addstr(row_y, 2, f"{metrics['step']:>6}")
            self.stdscr.addstr(row_y, 2 + col_width, f"{metrics['loss']:>8.4f}")
            self.stdscr.addstr(row_y, 2 + col_width * 2, f"{metrics['learning_rate']:>8.6f}")
            self.stdscr.addstr(row_y, 2 + col_width * 3, f"{metrics['gradient_norm']:>8.2f}")

    def _draw_progress_bar(self, y: int, x: int, current: int, total: int, message: str):
        """Draw a progress bar with message and time estimation"""
        width = 40
        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)
        percentage = (current / total * 100) if total > 0 else 0
        
        self.stdscr.addstr(y, x, f"{message}")
        self.stdscr.addstr(y + 1, x, f"[{bar}] {percentage:>3.0f}%")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into hours:minutes:seconds"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"