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
Version: 0.3.2
Title: multiple curses clean() in the modules
What went wrong?: multiple curses clean executes and interupts training
What was fixed?: curses.endwin() is called exactly once. final results are saved before cleanup. terminal is properly stored before running inference. error handling still properly cleans up the terminal
Version: 0.3.3
Title: color code gradient gauge display
What went wrong?: alerts are not visible and inferrable
What was fixed?: adds more granular warning levels, uses blinking text for crtical alerts, adds a helpful message of warning when gradients are too high, uses consistent color codings with the rest of the dashboard
Version: 0.3.4
Title: show inference within dashboard
What went wrong?: dashboard ends after training to show the inference
What was fixed?: keep the dashboard running during inference, give users time to see the results, display the inference results in the dashboard, clean up properly after showing all information. The Changes made in the making test_model return results instead of printing. Increasing results to the dashboard display. Increasing the viewing time before cleanup. Maintaining dashboard until arell the results are shown
Version: 0.3.5
Title: visual loss, val metrics, interactive keyboard
What went wrong?: loss was not easy to infer and no keyboard control during dashboard
What was fixed?: visual loss trend, clear validation metrics, interactive q and s controls
