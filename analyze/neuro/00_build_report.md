Read through the results in the `experiments/runs-neuro` directory and make an interactive html report `analyze/neuro/report.html`.

The report should show the main result (the metric as a function of iterations) for each run, labeling the run with the model / thinking effort it uses. You might need to read the copilot cli logs to find those (if you do, write a `metadata.json` file into each run's folder with the relevant info for the report for next time).

The `fmri-may27-run1` did not trim the story ends (all other runs trimmed 30 TRs off each end). Present the may27 run first separately from the rest and mention this difference. At the bottom of the report add some key details on why this is necessary with a plot or two by reading the notes in the `fmri-may27-run1/analysis/report.md`. Also note in the report that the `fmri-jun04-run1`
was prompted to read the results of all the runs before it when starting, so it had more information to work with than the other runs.
Make it so that the user can interactively see what methods were tried, and give a writeup based on the overall_results.csv descriptions on what methods were tried, and which worked / didn't.

Flag and exclude any iterations that used any training on data (including text pretraining), as this is explicitly disallowed.

Style everything including the code as a white-background, clean design.

At the top, show a summary plot that has all the datasets in one plot for trimmed runs and another for non-trimmed runs. The points for each run should be connected as a line.

At the bottom of the page, give detailed explanations on some of the best hand-engineered features, how they were implemented, and how they contributed to the results. Include code snippets and visualizations where relevant.

Check the rendering at the end and make sure that things are non-overlapping and readable, even when things are clicked. If not, make adjustments to the layout and spacing.
