import curses
import time
import numpy as np
from typing import Dict, List, Optional
from config import TrainingConfig

class TrainingDashboard:
    def __init__(self, stdscr, config: TrainingConfig):
        self.stdscr = stdscr
        self.config = config
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        self.metrics_history = []
        
    def update_metrics(self, metrics: Dict[str, float]):
        self.metrics_history.append(metrics)
        self._draw_dashboard()
        
    def _draw_speedometer(self, y: int, x: int, value: float, max_value: float, title: str):
        # Draw speedometer-like visualization
        width = 20
        filled = int((value / max_value) * width)
        self.stdscr.addstr(y, x, f"{title}: ")
        self.stdscr.addstr(y, x + len(title) + 2, "[")
        self.stdscr.addstr("=" * filled)
        self.stdscr.addstr(" " * (width - filled))
        self.stdscr.addstr(f"] {value:.4f}")
        
    def _draw_gradient_gauge(self, y: int, x: int, gradient_norm: float):
        """Draw gradient norm with warning colors"""
        self.stdscr.addstr(y, x, "Gradient Norm: ")
        if gradient_norm > 10.0:
            self.stdscr.addstr(f"{gradient_norm:.2f}", curses.color_pair(2))  # Red
        elif gradient_norm > 5.0:
            self.stdscr.addstr(f"{gradient_norm:.2f}", curses.color_pair(3))  # Yellow
        else:
            self.stdscr.addstr(f"{gradient_norm:.2f}", curses.color_pair(1))  # Green

    def _draw_early_stopping(self, y: int, x: int, steps_without_improvement: int, patience: int):
        """Draw early stopping progress"""
        progress = steps_without_improvement / patience
        width = 20
        filled = int(progress * width)
        
        self.stdscr.addstr(y, x, "Early Stopping: ")
        self.stdscr.addstr("[")
        if progress > 0.8:
            color = curses.color_pair(2)  # Red
        elif progress > 0.5:
            color = curses.color_pair(3)  # Yellow
        else:
            color = curses.color_pair(1)  # Green
        
        self.stdscr.addstr("=" * filled, color)
        self.stdscr.addstr(" " * (width - filled))
        self.stdscr.addstr(f"] {steps_without_improvement}/{patience}")

    def _draw_dashboard(self):
        self.stdscr.clear()
        
        # Title
        self.stdscr.addstr(0, 0, "Training Dashboard", curses.A_BOLD)
        
        # Current metrics
        if self.metrics_history:
            current_metrics = self.metrics_history[-1]
            
            # Training Loss Speedometer
            self._draw_speedometer(2, 0, current_metrics.get('loss', 0), 2.0, "Loss")
            
            # Learning Rate Indicator
            self._draw_speedometer(3, 0, current_metrics.get('learning_rate', 0), 1e-4, "LR")
            
            # Accuracy Gauge
            if 'eval_accuracy' in current_metrics:
                self._draw_speedometer(4, 0, current_metrics['eval_accuracy'], 1.0, "Accuracy")
            
            # F1 Score Gauge
            if 'eval_f1' in current_metrics:
                self._draw_speedometer(5, 0, current_metrics['eval_f1'], 1.0, "F1 Score")
            
            # Training Progress
            epoch = current_metrics.get('epoch', 0)
            self.stdscr.addstr(6, 0, f"Epoch: {epoch}")
            
            # GPU Memory Usage
            gpu_memory = current_metrics.get('gpu_memory', 0)
            self.stdscr.addstr(7, 0, f"GPU Memory: {gpu_memory:.2f} MB")
            
            # Gradient Norm
            self._draw_gradient_gauge(8, 0, current_metrics.get('gradient_norm', 0))
            
            # Early Stopping
            self._draw_early_stopping(9, 0, 
                current_metrics.get('steps_without_improvement', 0),
                self.config.early_stopping_patience)
        
        self.stdscr.refresh()