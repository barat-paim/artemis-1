class TrainingDashboard:
    # ... existing code ...
    
    def _draw_speedometer(self, window, x, y, value, max_value, title=""):
        """
        Draws a simple speedometer-style progress indicator
        
        Args:
            window: The curses window to draw on
            x, y: The coordinates to start drawing
            value: Current value to display
            max_value: Maximum value for scaling
            title: Optional title for the speedometer
        """
        width = 20  # width of the speedometer
        percentage = min(value / max_value, 1.0) if max_value > 0 else 0
        filled = int(width * percentage)
        
        if title:
            window.addstr(y, x, title)
            y += 1
            
        window.addstr(y, x, "[" + "=" * filled + " " * (width - filled) + "]")
        window.addstr(y + 1, x, f"{value:.2f}/{max_value:.2f}") 