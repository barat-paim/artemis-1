What went wrong?: error occured because we were calling update_metrics in the monitor without updating the the training metrics in dashboard
What was fixed?: store the new metrics in history, trigger a dashboard redraw
Version: 0.2.0
Title: Changes in this version
What went wrong?: Tracking the Training Metrics is Painful
What was fixed?: Set up a new Dashboard using Curses
Version: 0.2.1
Title: fix: dashboard improvements
What went wrong?: code misalignment
What was fixed?: speedometer, gradient gauge, early stopping in dashboard.py
Version: 0.3.0
Title: fix: replace print with dashboard statuses
What went wrong?: circular imports, dashboard status instead of print, functional bugs
What was fixed?: main.py creates both dashboard & monitor (which receives dashboard as a parameters)[D[D[D[Dtrainer.py uses monitor for logging
Version: 0.3.1
Title: dashboard includes evaluation entries, prevent keyError
What went wrong?: keyError stops training in between
What was fixed?: draw_eval_table has get() to to handle missing metrics
