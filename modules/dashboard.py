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

    #### section for setting up the colors ####
    def setup_colors(self):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
    #### section for setting the status message ####
    def set_status(self, message: str):
        self.status_message = message
        self._draw_dashboard()
        
    #### section for drawing the section header ####
    def _draw_section_header(self, y: int, x: int, title: str):
        self.stdscr.addstr(y, x, f"═══ {title} ", curses.color_pair(4) | curses.A_BOLD)
        remaining_width = self.max_x - (x + len(title) + 5)
        self.stdscr.addstr(y, x + len(title) + 5, "═" * remaining_width, curses.color_pair(4))
        
    #### section for saving the final results ####
    def cleanup(self):
        """Save final results"""
        self._save_final_results()
        
    #### section for saving the final results to results.md ####
    def _save_final_results(self):
        if not self.metrics_history:
            return
            
        final_metrics = self.metrics_history[-1]
        eval_metrics = next((m for m in reversed(self.metrics_history) 
                            if 'eval_accuracy' in m), {})
        
        with open(self.results_file, "w") as f:
            f.write("# Training Results\n\n")
            f.write("## Final Metrics\n")
            f.write(f"- Loss: {final_metrics.get('loss', 0):.4f}\n")
            f.write(f"- Accuracy: {eval_metrics.get('eval_accuracy', 0):.2%}\n")
            f.write(f"- F1 Score: {eval_metrics.get('eval_f1', 0):.2%}\n")
            f.write(f"- Training Time: {final_metrics.get('time_elapsed', 0)/60:.1f} minutes\n")
            
    #### DRAW DASHBOARD ####
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
            
            # Draw validation metrics
            if any('eval_accuracy' in m for m in self.metrics_history):
                val_y = status_y + 4
                self._draw_section_header(val_y, 0, "Validation Metrics")
                latest_eval = next((m for m in reversed(self.metrics_history) 
                                  if 'eval_accuracy' in m), {})
                
                self.stdscr.addstr(val_y + 1, 2, f"Accuracy: {latest_eval.get('eval_accuracy', 0):.2%}")
                self.stdscr.addstr(val_y + 2, 2, f"F1 Score: {latest_eval.get('eval_f1', 0):.2%}")
                
                # Draw loss trend plot
                loss_history = [m.get('loss', 0) for m in self.metrics_history[-10:]]
                plot = self._create_ascii_plot(loss_history)
                self._draw_section_header(val_y + 4, 0, "Loss Trend")
                for i, line in enumerate(plot):
                    self.stdscr.addstr(val_y + 5 + i, 2, line)
                
                # Draw keyboard shortcuts at the bottom
                self._draw_keyboard_shortcuts(self.max_y - 3, 2)
        
        self.stdscr.refresh()
        
    #### section for defining the dashboard elements ####
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics and redraw dashboard"""
        self.metrics_history.append(metrics)
        if metrics.get('is_eval_step', False):
            self.eval_history.append(metrics)
        self._draw_dashboard()

    def _draw_speedometer(self, y: int, x: int, value: float, max_value: float, label: str):
        """Draw a speedometer-style progress bar"""
        width = 30
        # invert the fill ratio since lower loss is better
        normalized_value = max(0, min(1, value / max_value))
        inverted_fill = 1 - normalized_value
        filled = int(width * inverted_fill)
        bar = "█" * filled + "░" * (width - filled)

        # show percentage improvement rather than raw percentage
        improvement = (1 - normalized_value) * 100
        self.stdscr.addstr(y, x, f"{label:<10}")
        self.stdscr.addstr(y, x + 10, f"[{bar}]")
        self.stdscr.addstr(y, x + width + 12, f"{value:.4f} ({improvement:.1f}% improved)")

    def _draw_gradient_gauge(self, y: int, x: int, gradient_norm: float):
        """Draw a gauge showing gradient norm status with enhanced color coding"""
        if gradient_norm < 1.0:
            status = "GOOD"
            color = curses.color_pair(1)  # Green
        elif gradient_norm < 5.0:
            status = "WARN"
            color = curses.color_pair(3)  # Yellow
        elif gradient_norm < 10.0:
            status = "HIGH"
            color = curses.color_pair(2)  # Red
        else:
            status = "ALERT"
            color = curses.color_pair(2) | curses.A_BLINK  # Blinking Red
        
        # Draw the label
        self.stdscr.addstr(y, x, f"Gradient Norm: {gradient_norm:.2f} ")
        
        # Draw the status with color
        self.stdscr.addstr(f"[{status}]", color | curses.A_BOLD)
        
        # Add warning message for high gradients
        if gradient_norm >= 5.0:
            warning = " (Consider reducing learning rate)"
            self.stdscr.addstr(warning, curses.color_pair(3))

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
            # Use get() with default values to handle missing metrics
            self.stdscr.addstr(row_y, 2, f"{metrics.get('step', 0):>6}")
            self.stdscr.addstr(row_y, 2 + col_width, f"{metrics.get('loss', 0):>8.4f}")
            self.stdscr.addstr(row_y, 2 + col_width * 2, f"{metrics.get('learning_rate', 0):>8.6f}")
            self.stdscr.addstr(row_y, 2 + col_width * 3, f"{metrics.get('gradient_norm', 0):>8.2f}")

    def _draw_progress_bar(self, y: int, x: int, current: int, total: int, message: str):
        """Draw a progress bar with message and time estimation"""
        width = 40
        # Ensure we have valid values
        current = current if current is not None else 0
        total = total if total is not None else 1
        
        filled = int(width * (current / total)) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)
        percentage = (current / total * 100) if total > 0 else 0
        
        # Update message with progress
        progress_msg = f"{message} ({current}/{total} steps)"
        self.stdscr.addstr(y, x, progress_msg)
        self.stdscr.addstr(y + 1, x, f"[{bar}] {percentage:>3.0f}%")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into hours:minutes:seconds"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _draw_inference_results(self, y: int, x: int, results: List[Dict]):
        """Draw inference results in dashboard"""
        self._draw_section_header(y, 0, "Inference Results")
        for i, result in enumerate(results):
            text = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
            # Try prediction first, fall back to sentiment if prediction not found
            pred = result.get('prediction', result.get('sentiment', 'N/A'))
            conf = result.get('confidence', 0.0) * 100  # Convert to percentage
            
            self.stdscr.addstr(y + i + 1, x, f"{text}")
            self.stdscr.addstr(y + i + 1, x + 55, f"{pred} ({conf:.1f}%)", 
                              curses.color_pair(1) if conf > 70 else curses.color_pair(3))

    def _create_ascii_plot(self, values: List[float], width: int = 30, height: int = 5) -> List[str]:
        """Create an ASCII plot of the loss trend"""
        if not values:
            return []
            
        # Normalize values to fit height
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        normalized = [(h - min_val) / range_val * (height - 1) for h in values]
        
        # Create the plot
        plot = []
        for h in range(height - 1, -1, -1):
            line = ""
            for n in normalized:
                if n >= h:
                    line += "█"
                else:
                    line += "░"
            plot.append(line)
        
        return plot

    def _draw_keyboard_shortcuts(self, y: int, x: int):
        """Draw keyboard shortcuts section"""
        self.stdscr.addstr(y, x, "Keyboard Shortcuts:", curses.A_BOLD)
        self.stdscr.addstr(y + 1, x, "q: Quit  |  s: Save Checkpoint  |  p: Pause/Resume")

    def _show_checkpoint_info(self, checkpoint_path: Path, latest_path: Path):
        """Show checkpoint save locations"""
        save_y = self.max_y - 5  # Show near bottom of screen
        self.stdscr.addstr(save_y, 2, "Checkpoints saved to:", curses.A_BOLD)
        self.stdscr.addstr(save_y + 1, 4, f"Step checkpoint: {checkpoint_path}")
        self.stdscr.addstr(save_y + 2, 4, f"Latest checkpoint: {latest_path}")