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
        
        if self.metrics_history:
            current_metrics = self.metrics_history[-1]
            
            # Training Metrics Section
            self._draw_section_header(2, 0, "Training Progress")
            self._draw_speedometer(4, 2, current_metrics.get('loss', 0), 2.0, "Loss")
            self._draw_speedometer(5, 2, current_metrics.get('learning_rate', 0), 1e-4, "LR")
            self._draw_gradient_gauge(6, 2, current_metrics.get('gradient_norm', 0))
            
            # Evaluation Metrics Section
            self._draw_section_header(8, 0, "Evaluation Metrics")
            if 'eval_accuracy' in current_metrics:
                self._draw_speedometer(10, 2, current_metrics['eval_accuracy'], 1.0, "Accuracy")
            if 'eval_f1' in current_metrics:
                self._draw_speedometer(11, 2, current_metrics['eval_f1'], 1.0, "F1 Score")
            
            # System Metrics Section
            self._draw_section_header(13, 0, "System Status")
            epoch = current_metrics.get('epoch', 0)
            gpu_mem = current_metrics.get('gpu_memory', 0)
            self.stdscr.addstr(15, 2, f"Epoch: {epoch}")
            self.stdscr.addstr(15, 20, f"GPU Memory: {gpu_mem:.0f}MB")
            self._draw_early_stopping(16, 2, 
                current_metrics.get('steps_without_improvement', 0),
                self.config.early_stopping_patience)
            
            # Status Message
            if self.status_message:
                self._draw_section_header(18, 0, "Status")
                self.stdscr.addstr(19, 2, self.status_message)
        
        self.stdscr.refresh()