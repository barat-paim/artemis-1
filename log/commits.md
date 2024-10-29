What went wrong?: error occured because we were calling update_metrics in the monitor without updating the the training metrics in dashboard
What was fixed?: store the new metrics in history, trigger a dashboard redraw
