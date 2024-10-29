What went wrong?: error occured because we were calling update_metrics in the monitor without updating the the training metrics in dashboard
What was fixed?: store the new metrics in history, trigger a dashboard redraw
Version: 0.2.0
Title: Changes in this version
What went wrong?: Tracking the Training Metrics is Painful
What was fixed?: Set up a new Dashboard using Curses
